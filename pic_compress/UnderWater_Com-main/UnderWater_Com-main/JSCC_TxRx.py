import json
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import matlab.engine
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils import AudioTxRuntimeConfig, run_audio_txrx_pipeline


METHOD_NAME = "JSCC传输"

PROJECT_ROOT = None
_probe = ROOT_DIR
while _probe != _probe.parent:
    if (_probe / "pic_compress").is_dir():
        PROJECT_ROOT = _probe
        break
    _probe = _probe.parent
if PROJECT_ROOT is None:
    PROJECT_ROOT = ROOT_DIR.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PIC_COMPRESS_IMPORT_ERRORS = []

try:
    from pic_compress.model import CompressAIJSCCModel  # noqa: E402
except Exception as exc:
    CompressAIJSCCModel = None
    PIC_COMPRESS_IMPORT_ERRORS.append(f"pic_compress.model: {exc}")

try:
    from pic_compress.test_real_channel import MultiBitSTEQuantizer, split_quant_bits  # noqa: E402
except Exception:
    try:
        from pic_compress.quantizers import MultiBitSTEQuantizer  # noqa: E402

        def split_quant_bits(total_bits, preferred_base_bits=3, min_res_bits=1):
            total_bits = int(max(1, total_bits))
            preferred_base_bits = int(max(1, preferred_base_bits))
            min_res_bits = int(max(0, min_res_bits))
            if total_bits <= preferred_base_bits:
                return total_bits, 0
            base_bits = min(preferred_base_bits, total_bits - min_res_bits)
            base_bits = max(1, min(total_bits, base_bits))
            res_bits = max(0, total_bits - base_bits)
            return int(base_bits), int(res_bits)
    except Exception as exc:
        PIC_COMPRESS_IMPORT_ERRORS.append(f"pic_compress.quantizers: {exc}")

        class MultiBitSTEQuantizer:  # Fallback to avoid import-time crash on missing dependency.
            def __init__(self, bits=4, clip_val=1.0, mode="tanh", compand_mu=6.0):
                self.bits = int(bits)
                self.clip_val = float(clip_val)

            def quantize_with_indices(self, x):
                levels = _build_tanh_levels(self.bits, device=x.device, dtype=x.dtype) * self.clip_val
                x_clip = x.clamp(-self.clip_val, self.clip_val)
                x_flat = x_clip.reshape(-1)
                dist = torch.abs(x_flat[:, None] - levels[None, :])
                idx = torch.argmin(dist, dim=1)
                q = levels[idx].reshape_as(x)
                return q, idx.reshape_as(x)

        def split_quant_bits(total_bits, preferred_base_bits=3, min_res_bits=1):
            total_bits = int(max(1, total_bits))
            preferred_base_bits = int(max(1, preferred_base_bits))
            min_res_bits = int(max(0, min_res_bits))
            if total_bits <= preferred_base_bits:
                return total_bits, 0
            base_bits = min(preferred_base_bits, total_bits - min_res_bits)
            base_bits = max(1, min(total_bits, base_bits))
            res_bits = max(0, total_bits - base_bits)
            return int(base_bits), int(res_bits)

try:
    from pic_compress.utils import compute_psnr, compute_ssim  # noqa: E402
except Exception:
    def compute_psnr(img1_t, img2_t):
        x = torch.clamp(img1_t, 0.0, 1.0)
        y = torch.clamp(img2_t, 0.0, 1.0)
        mse = torch.mean((x - y) ** 2).item()
        if mse <= 1e-12:
            return float("inf")
        return float(10.0 * math.log10(1.0 / mse))

    def compute_ssim(img1_t, img2_t):
        # Lightweight fallback for environments missing pic_compress.utils.
        x = torch.clamp(img1_t, 0.0, 1.0)
        y = torch.clamp(img2_t, 0.0, 1.0)
        c1 = (0.01 ** 2)
        c2 = (0.03 ** 2)
        mu_x = torch.mean(x)
        mu_y = torch.mean(y)
        sigma_x = torch.mean((x - mu_x) ** 2)
        sigma_y = torch.mean((y - mu_y) ** 2)
        sigma_xy = torch.mean((x - mu_x) * (y - mu_y))
        num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        den = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
        return float((num / torch.clamp(den, min=1e-12)).item())


MODEL = None
DEVICE = None
ENG = None
LAST_PHY_STATS = {}

IMAGE_SIZE = (256, 256)
QUANT_BITS = 4
DEFAULT_SNR_DB = 10.0

TX_BITSTREAM_PATH = ROOT_DIR / "tx_bitstream_jscc.txt"
META_PATH = ROOT_DIR / "savedata" / "jscc_meta.json"
TX_WAV_PATH = ROOT_DIR / "savedata" / "tx_jscc.wav"
DEFAULT_RX_WAV_PATH = ROOT_DIR / "savedata" / "rx_jscc.wav"
DEFAULT_RX_IMAGE_PATH = ROOT_DIR / "rx_output_jscc.png"
RECON_ARCHIVE_DIR = ROOT_DIR / "savedata" / "reconstructions"

TX_MONITOR_CHUNK_SIZE = 1024
RX_RECORD_BLOCKSIZE = 1024
RECORD_PREROLL_SEC = 1.5
RECORD_MAX_TAIL_SEC = 6.0
RECORD_SILENCE_HOLD_SEC = 1.2
RECORD_NOISE_CALIBRATION_SEC = 0.8
RECORD_START_TIMEOUT_SEC = 5.0
AUDIO_RUNTIME_CFG = AudioTxRuntimeConfig(
    tx_monitor_chunk_size=TX_MONITOR_CHUNK_SIZE,
    rx_record_blocksize=RX_RECORD_BLOCKSIZE,
    record_preroll_sec=RECORD_PREROLL_SEC,
    record_max_tail_sec=RECORD_MAX_TAIL_SEC,
    record_silence_hold_sec=RECORD_SILENCE_HOLD_SEC,
    record_noise_calibration_sec=RECORD_NOISE_CALIBRATION_SEC,
    record_start_timeout_sec=RECORD_START_TIMEOUT_SEC,
)


def _log(log_callback, text):
    msg = f"[{METHOD_NAME}] {text}"
    print(msg)
    if callable(log_callback):
        try:
            log_callback(msg)
        except Exception:
            pass


def _ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _resolve_ckpt_path():
    candidates = [
        PROJECT_ROOT / "pic_compress" / "checkpoints" / "0.pth",
        PROJECT_ROOT / "pic_compress" / "jscc_swin_lpips_qb4_im256_step600.pth",
        PROJECT_ROOT / "pic_compress" / "checkpoints" / "jscc_swin_lpips_qb4_im256_step200.pth",
        PROJECT_ROOT / "pic_compress" / "jscc_swin_lpips_qb4_im256_step200.pth",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "未找到 jscc_swin_lpips_qb4_im256_step200.pth 或 jscc_swin_lpips_qb4_im256_step600.pth，"
        "请放在 pic_compress/ 或 pic_compress/checkpoints/ 下。"
    )


def _build_tanh_levels(bits, device, dtype=torch.float32):
    n_levels = 1 << int(bits)
    base = torch.linspace(-1.0, 1.0, steps=n_levels, device=device, dtype=dtype)
    return torch.tanh(1.5 * base) / math.tanh(1.5)


def _gray_encode_indices(indices_np, bits):
    indices_np = np.asarray(indices_np, dtype=np.int64).reshape(-1)
    gray = indices_np ^ (indices_np >> 1)
    shifts = np.arange(bits - 1, -1, -1, dtype=np.int64)
    out = ((gray[:, None] >> shifts[None, :]) & 1).astype(np.uint8)
    return out.reshape(-1)


def _gray_decode_bits(bits_np, bits):
    b = np.asarray(bits_np, dtype=np.uint8).reshape(-1)
    total = (b.size // bits) * bits
    if total <= 0:
        return np.zeros((0,), dtype=np.int64)
    b = b[:total].reshape(-1, bits).astype(np.int64)
    shifts = np.arange(bits - 1, -1, -1, dtype=np.int64)
    gray = np.sum(b * (1 << shifts[None, :]), axis=1)
    binary = gray.copy()
    shift = 1
    while shift < (1 << bits):
        binary ^= (binary >> shift)
        shift <<= 1
    return binary


def _apply_center_frequency(center_frequency_hz, log_callback=None):
    if center_frequency_hz is None:
        return
    if ENG is None or not hasattr(ENG, "set_center_frequency"):
        return
    applied_fc = ENG.set_center_frequency(float(center_frequency_hz))
    _log(log_callback, f"中心频率设置为 {applied_fc:.1f} Hz")


def _apply_phy_params(phy_params, log_callback=None):
    if not phy_params:
        return
    if ENG is None or not hasattr(ENG, "set_phy_params"):
        return
    applied = ENG.set_phy_params(**dict(phy_params))
    _log(
        log_callback,
        f"水声参数: rolloff={applied.get('rolloff', '-')}, "
        f"pilot_amp={applied.get('pilot_amp', '-')}, sps={applied.get('sps', '-')}",
    )


def _load_image_tensor(img_path):
    raw = cv2.imread(str(img_path))
    if raw is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")
    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    rgb = _resize_center_crop_rgb(rgb, IMAGE_SIZE)
    x = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
    return x.unsqueeze(0).to(DEVICE)


def _resize_center_crop_rgb(rgb, size):
    target_w = int(size[0])
    target_h = int(size[1])
    h, w = rgb.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("输入图像尺寸非法")

    # Match eval script behavior: Resize(shorter_side=target) + CenterCrop(target,target)
    scale = max(target_w / float(w), target_h / float(h))
    new_w = max(target_w, int(round(w * scale)))
    new_h = max(target_h, int(round(h * scale)))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    off_x = max(0, (new_w - target_w) // 2)
    off_y = max(0, (new_h - target_h) // 2)
    cropped = resized[off_y : off_y + target_h, off_x : off_x + target_w]
    if cropped.shape[0] != target_h or cropped.shape[1] != target_w:
        cropped = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return cropped


def _encode_image_to_latent(x):
    snr = torch.full((x.shape[0], 1), float(DEFAULT_SNR_DB), device=DEVICE)
    with torch.no_grad():
        y = MODEL.g_a(x)
        y = MODEL.front_adapter(y)
        tx_feedback = MODEL._build_tx_feedback_snr(snr)
        s = MODEL.jscc_encoder((y, tx_feedback))
        s = MODEL.power_constraint(s)
        s, n_pilot = MODEL._insert_pilots(s)
    return s, int(n_pilot)


def _quantize_to_bits(s):
    base_bits, res_bits = split_quant_bits(int(QUANT_BITS), preferred_base_bits=3, min_res_bits=1)
    res_scale = float(2 ** int(base_bits))

    quantizer_base = MultiBitSTEQuantizer(bits=base_bits, clip_val=1.0, mode="tanh", compand_mu=6.0)
    base_q, base_idx = quantizer_base.quantize_with_indices(s)
    base_bits_np = _gray_encode_indices(base_idx.detach().cpu().numpy(), base_bits)

    if res_bits > 0:
        residual = torch.tanh(s) - base_q
        quantizer_res = MultiBitSTEQuantizer(bits=res_bits, clip_val=1.0, mode="tanh", compand_mu=6.0)
        _, res_idx = quantizer_res.quantize_with_indices(residual * res_scale)
        res_bits_np = _gray_encode_indices(res_idx.detach().cpu().numpy(), res_bits)
        bits_np = np.concatenate([base_bits_np, res_bits_np], axis=0)
    else:
        bits_np = base_bits_np

    meta = {
        "shape": list(s.shape),
        "quant_bits": int(QUANT_BITS),
        "base_bits": int(base_bits),
        "res_bits": int(res_bits),
        "n_elements": int(np.prod(s.shape)),
        "snr_db": float(DEFAULT_SNR_DB),
    }
    return bits_np, meta


def _dequantize_from_bits(rx_bits_np, meta):
    quant_bits = int(meta.get("quant_bits", QUANT_BITS))
    base_bits = int(meta.get("base_bits", quant_bits))
    res_bits = int(meta.get("res_bits", max(0, quant_bits - base_bits)))
    n_elements = int(meta.get("n_elements", 0))
    shape = tuple(int(v) for v in meta["shape"])
    if n_elements <= 0:
        n_elements = int(np.prod(shape))
    expected_bits = int(n_elements * (base_bits + res_bits))
    b = np.asarray(rx_bits_np, dtype=np.uint8).reshape(-1)
    if b.size < expected_bits:
        pad = np.zeros(expected_bits - b.size, dtype=np.uint8)
        b = np.concatenate([b, pad], axis=0)
    else:
        b = b[:expected_bits]

    base_total = int(n_elements * base_bits)
    base_bits_np = b[:base_total]
    base_idx = _gray_decode_bits(base_bits_np, base_bits)
    base_levels = _build_tanh_levels(base_bits, device=DEVICE)
    base_idx_t = torch.from_numpy(base_idx).to(DEVICE).long()
    s_base = base_levels[base_idx_t]

    if res_bits > 0:
        res_total = int(n_elements * res_bits)
        res_bits_np = b[base_total: base_total + res_total]
        res_idx = _gray_decode_bits(res_bits_np, res_bits)
        res_levels = _build_tanh_levels(res_bits, device=DEVICE)
        res_idx_t = torch.from_numpy(res_idx).to(DEVICE).long()
        s_res = res_levels[res_idx_t] / float(2 ** base_bits)
        s_rx = s_base + s_res
    else:
        s_rx = s_base

    s_rx = s_rx.reshape(shape)
    return s_rx


def _decode_latent_to_image(s_rx, n_pilot):
    with torch.no_grad():
        s_rx = MODEL.semantic_equalizer(s_rx)
        snr_hat = MODEL._estimate_snr_hat(s_rx, int(n_pilot))
        y_hat = MODEL.jscc_decoder((s_rx, snr_hat))
        y_hat = MODEL.latent_refiner(y_hat)
        x_hat = MODEL.g_s(y_hat)
        x_hat = MODEL.detail_refiner(x_hat)
        x_hat = x_hat.clamp(0.0, 1.0)
    return x_hat


def _save_tensor_image(x_hat, save_img_path):
    arr = x_hat[0].detach().cpu().permute(1, 2, 0).numpy()
    img = (arr * 255.0 + 0.5).astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_img_path), bgr)


def _archive_reconstruction_image(saved_img_path):
    src = Path(saved_img_path)
    if not src.is_file():
        return None
    RECON_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dst = RECON_ARCHIVE_DIR / f"{src.stem}_{stamp}{src.suffix or '.png'}"
    shutil.copy2(src, dst)
    return str(dst)


def init_system():
    global MODEL, DEVICE, ENG
    if CompressAIJSCCModel is None:
        details = "; ".join(PIC_COMPRESS_IMPORT_ERRORS) if PIC_COMPRESS_IMPORT_ERRORS else "unknown import error"
        raise RuntimeError(
            "JSCC 模型依赖不可用：无法导入 pic_compress.model。"
            f" 详情: {details}"
        )
    if ENG is None:
        ENG = matlab.engine.start_matlab()
    if DEVICE is None:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if MODEL is None:
        MODEL = CompressAIJSCCModel(
            N=128,
            M=192,
            jscc_hidden=128,
            channel_out=16,
            use_proxy_channel=False,
            use_lrformer_front=True,
            use_swin=True,
            quant_bits=QUANT_BITS,
            quant_mode="tanh",
            quant_compand_mu=6.0,
        ).to(DEVICE)
        ckpt_path = _resolve_ckpt_path()
        state = torch.load(str(ckpt_path), map_location=DEVICE)
        model_state = MODEL.state_dict()
        filtered = {k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape}
        MODEL.load_state_dict(filtered, strict=False)
        MODEL.eval()
        print(f"[{METHOD_NAME}] loaded checkpoint: {ckpt_path} (keys={len(filtered)})")


def estimate_tx_bits(img_path):
    init_system()
    x = _load_image_tensor(img_path)
    s, _ = _encode_image_to_latent(x)
    return int(s.numel() * QUANT_BITS)


@torch.no_grad()
def Tx(
    img_path,
    rx_wav_path=str(DEFAULT_RX_WAV_PATH),
    ams22_device_index=6,
    rx_channels=1,
    rx_samplerate=64000,
    center_frequency_hz=8000.0,
    phy_params=None,
    monitor_callback=None,
    log_callback=None,
    force_offline_loopback=True,
):
    init_system()
    _ensure_parent(TX_BITSTREAM_PATH)
    _ensure_parent(TX_WAV_PATH)
    _ensure_parent(META_PATH)
    _ensure_parent(rx_wav_path)

    x = _load_image_tensor(img_path)
    s, n_pilot = _encode_image_to_latent(x)
    tx_bits_np, meta = _quantize_to_bits(s)
    meta["n_pilot"] = int(n_pilot)

    with open(TX_BITSTREAM_PATH, "w", encoding="utf-8") as f:
        f.write("".join("1" if int(b) else "0" for b in tx_bits_np))
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    _log(log_callback, f"发送比特流长度: {len(tx_bits_np)} bits")
    _apply_phy_params(phy_params, log_callback=log_callback)
    _apply_center_frequency(center_frequency_hz, log_callback=log_callback)
    ENG.Copy_2_of_main_GenSignal(str(TX_BITSTREAM_PATH), str(TX_WAV_PATH), nargout=0)
    _log(log_callback, f"调制完成，发送波形已保存到: {TX_WAV_PATH}")

    return run_audio_txrx_pipeline(
        method_name=METHOD_NAME,
        wav_path=str(TX_WAV_PATH),
        rx_wav_path=str(rx_wav_path),
        tx_bits=int(len(tx_bits_np)),
        tx_bitstream_path=str(TX_BITSTREAM_PATH),
        ams22_device_index=ams22_device_index,
        rx_channels=rx_channels,
        rx_samplerate=rx_samplerate,
        force_offline_loopback=bool(force_offline_loopback),
        monitor_callback=monitor_callback,
        log_callback=log_callback,
        runtime_cfg=AUDIO_RUNTIME_CFG,
    )


@torch.no_grad()
def Rx(
    rx_bits_path,
    save_img_path=str(DEFAULT_RX_IMAGE_PATH),
    rx_wav_path=str(DEFAULT_RX_WAV_PATH),
    center_frequency_hz=8000.0,
    phy_params=None,
    log_callback=None,
    **kwargs,
):
    global LAST_PHY_STATS
    init_system()
    _ensure_parent(rx_bits_path)
    _ensure_parent(save_img_path)

    if not META_PATH.is_file():
        raise FileNotFoundError(f"未找到元数据文件: {META_PATH}")

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    _apply_phy_params(phy_params, log_callback=log_callback)
    _apply_center_frequency(center_frequency_hz, log_callback=log_callback)
    detect_stats = ENG.Copy_2_of_main_DetecSignal(str(rx_wav_path), str(TX_BITSTREAM_PATH), str(rx_bits_path), nargout=1)
    LAST_PHY_STATS = dict(detect_stats) if isinstance(detect_stats, dict) else {}
    if LAST_PHY_STATS:
        _log(
            log_callback,
            f"PHY诊断: ber={LAST_PHY_STATS.get('ber', '-')}, "
            f"sync_peak={LAST_PHY_STATS.get('sync_peak', '-')}, "
            f"rx_rms={LAST_PHY_STATS.get('rx_passband_rms', '-')}",
        )

    with open(rx_bits_path, "r", encoding="utf-8") as f:
        rx_bits_str = "".join(ch for ch in f.read().strip() if ch in "01")
    rx_bits_np = np.fromiter((1 if ch == "1" else 0 for ch in rx_bits_str), dtype=np.uint8)
    _log(log_callback, f"接收比特流长度: {len(rx_bits_np)} bits")

    s_rx = _dequantize_from_bits(rx_bits_np, meta)
    x_hat = _decode_latent_to_image(s_rx, n_pilot=int(meta.get("n_pilot", 16)))
    _save_tensor_image(x_hat, save_img_path)
    _log(log_callback, f"重建图像已保存: {save_img_path}")
    archived_path = _archive_reconstruction_image(save_img_path)
    if archived_path:
        _log(log_callback, f"重建图像归档保存: {archived_path}")
    return str(save_img_path)


def calc_metrics_and_show(
    img_path1,
    img_path2,
    size=IMAGE_SIZE,
    bitstream_path=str(TX_BITSTREAM_PATH),
    **kwargs,
):
    img1 = cv2.imread(str(img_path1))
    img2 = cv2.imread(str(img_path2))
    if img1 is None or img2 is None:
        raise ValueError("图像路径错误，请检查路径！")

    size_tuple = (int(size[0]), int(size[1]))
    img1 = _resize_center_crop_rgb(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), size_tuple)
    img2 = _resize_center_crop_rgb(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), size_tuple)
    img1_t = torch.from_numpy(img1.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    img2_t = torch.from_numpy(img2.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    psnr = float(compute_psnr(img1_t, img2_t))
    ssim = float(compute_ssim(img1_t, img2_t))

    original_bits = int(img1.size * 8)
    try:
        with open(bitstream_path, "r", encoding="utf-8") as f:
            compressed_bits = sum(1 for ch in f.read() if ch in "01")
    except Exception:
        compressed_bits = int(np.prod([1, 16, 16, 16]) * QUANT_BITS)

    compression_ratio = 1.0 - compressed_bits / float(max(1, original_bits))
    return float(psnr), float(ssim), float(compression_ratio)


if __name__ == "__main__":
    init_system()
