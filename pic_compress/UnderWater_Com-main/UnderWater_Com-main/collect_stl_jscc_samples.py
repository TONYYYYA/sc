import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image

try:
    from torchvision import datasets
except Exception:
    datasets = None

import JSCC_TxRx as J


class MinimalSTL10TestDataset:
    """Lightweight STL10 test loader: only test_X.bin and test_y.bin are required."""

    def __init__(self, root):
        root = Path(root)
        bin_dir = root / "stl10_binary"
        x_path = bin_dir / "test_X.bin"
        y_path = bin_dir / "test_y.bin"
        if not x_path.is_file() or not y_path.is_file():
            raise FileNotFoundError(
                f"Missing STL10 binary files: {x_path} and {y_path}"
            )
        self._x = np.memmap(str(x_path), dtype=np.uint8, mode="r")
        self._y = np.memmap(str(y_path), dtype=np.uint8, mode="r")
        if self._x.size % (3 * 96 * 96) != 0:
            raise RuntimeError(f"Unexpected test_X.bin size: {self._x.size}")
        n = self._x.size // (3 * 96 * 96)
        if self._y.size < n:
            raise RuntimeError(
                f"test_y.bin labels are insufficient: labels={self._y.size}, images={n}"
            )
        self._n = int(n)
        self._x = self._x.reshape(self._n, 3, 96, 96)

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        idx = int(idx)
        if idx < 0 or idx >= self._n:
            raise IndexError(idx)
        chw = np.asarray(self._x[idx], dtype=np.uint8)
        # STL10 binary stores each channel in column-major order, so swap H/W.
        hwc = np.transpose(chw, (2, 1, 0))
        img = Image.fromarray(hwc, mode="RGB")
        label = int(self._y[idx]) - 1
        if label < 0:
            label = 0
        return img, label


def _ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _iso_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _read_bits(path):
    p = Path(path)
    if not p.is_file():
        return ""
    txt = p.read_text(encoding="utf-8", errors="ignore")
    return "".join(ch for ch in txt if ch in "01")


def _compute_ber(tx_bits_path, rx_bits_path):
    tx_bits = _read_bits(tx_bits_path)
    rx_bits = _read_bits(rx_bits_path)
    n = min(len(tx_bits), len(rx_bits))
    if n <= 0:
        return None
    err = sum(1 for a, b in zip(tx_bits[:n], rx_bits[:n]) if a != b)
    return float(err) / float(n)


def _payload_bits(path):
    bits = _read_bits(path)
    if not bits:
        return None
    return int(len(bits))


def _probe_wav(wav_path):
    p = Path(wav_path)
    if not p.is_file():
        return {}
    try:
        data, fs = sf.read(str(p), dtype="float32")
    except Exception:
        return {}
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr[:, 0]
    if arr.size == 0:
        return {}
    rms = float(np.sqrt(np.mean(np.square(arr)) + 1e-12))
    peak = float(np.max(np.abs(arr)))
    return {
        "rx_samplerate": int(fs),
        "rx_duration_s": float(arr.size / max(1, int(fs))),
        "rx_rms": rms,
        "rx_peak": peak,
    }


def _extract_estimated_snr(phy):
    for key in (
        "snr_est_db",
        "estimated_snr_db",
        "estimated_snr",
        "snr_db",
        "snr",
    ):
        v = _safe_float(phy.get(key))
        if v is not None:
            return v
    return None


def _load_stl_test_dataset(stl_root):
    stl_root = Path(stl_root)
    if datasets is not None:
        try:
            return datasets.STL10(root=str(stl_root), split="test", download=False)
        except Exception:
            pass
    return MinimalSTL10TestDataset(stl_root)


def _append_jsonl(path, payload):
    _ensure_parent(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _append_csv(path, payload, fieldnames):
    _ensure_parent(path)
    file_exists = Path(path).is_file()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        row = {k: payload.get(k, None) for k in fieldnames}
        writer.writerow(row)


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Collect STL10 JSCC Tx/Rx records with audio, bitstreams, and diagnostics."
    )
    parser.add_argument("--stl_root", type=str, default=str(root.parent.parent / "data"))
    parser.add_argument("--output_dir", type=str, default=str(root / "savedata" / "stl_jscc_collect"))
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force_offline_loopback", action="store_true")
    parser.add_argument("--input_device_idx", type=int, default=None)
    parser.add_argument("--output_device_idx", type=int, default=None)
    parser.add_argument("--rx_channels", type=int, default=1)
    parser.add_argument("--sample_rate", type=int, default=64000)
    parser.add_argument("--center_freq", type=float, default=8000.0)
    parser.add_argument("--tx_gain", type=float, default=None, help="Manually logged tx gain value.")
    parser.add_argument("--distance_m", type=float, default=None, help="Manually logged Tx-Rx distance.")
    parser.add_argument("--depth_m", type=float, default=None, help="Manually logged hydrophone depth.")
    parser.add_argument("--notes", type=str, default="", help="Free-form session/sample note.")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "samples.jsonl"
    csv_path = out_dir / "samples.csv"

    dataset = _load_stl_test_dataset(args.stl_root)
    total = len(dataset)
    if args.num_samples > total:
        raise ValueError(f"num_samples={args.num_samples} exceeds STL10 test size={total}")

    rng = np.random.default_rng(args.seed)
    chosen_idx = rng.choice(total, size=args.num_samples, replace=False).tolist()

    csv_fields = [
        "sample_name",
        "timestamp",
        "stl_index",
        "rx_rms",
        "sync_peak",
        "estimated_snr",
        "ber",
        "payload_bits",
        "input_device_idx",
        "output_device_idx",
        "tx_gain",
        "distance_m",
        "depth_m",
        "center_freq",
        "sample_rate",
        "notes",
        "transmission_mode",
        "tx_wav_path",
        "rx_wav_path",
        "tx_bits_path",
        "rx_bits_path",
        "recon_path",
        "source_image_path",
        "status",
        "error",
    ]

    print(f"[INFO] STL10 test total={total}, selected={args.num_samples}, seed={args.seed}")
    print(f"[INFO] output_dir={out_dir}")

    for i, stl_idx in enumerate(chosen_idx):
        sample_name = f"stlx{i}"
        ts = _iso_timestamp()
        sample_dir = samples_dir / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        source_image_path = images_dir / f"{sample_name}.png"
        tx_wav_dst = sample_dir / f"{sample_name}_tx.wav"
        rx_wav_path = sample_dir / f"{sample_name}_rx.wav"
        tx_bits_dst = sample_dir / f"{sample_name}_tx_bits.txt"
        rx_bits_path = sample_dir / f"{sample_name}_rx_bits.txt"
        recon_path = sample_dir / f"{sample_name}_recon.png"

        record = {
            "sample_name": sample_name,
            "timestamp": ts,
            "stl_index": int(stl_idx),
            "rx_rms": None,
            "sync_peak": None,
            "estimated_snr": None,
            "ber": None,
            "payload_bits": None,
            "input_device_idx": args.input_device_idx,
            "output_device_idx": args.output_device_idx,
            "tx_gain": args.tx_gain,
            "distance_m": args.distance_m,
            "depth_m": args.depth_m,
            "center_freq": float(args.center_freq),
            "sample_rate": int(args.sample_rate),
            "notes": args.notes,
            "transmission_mode": None,
            "tx_wav_path": str(tx_wav_dst),
            "rx_wav_path": str(rx_wav_path),
            "tx_bits_path": str(tx_bits_dst),
            "rx_bits_path": str(rx_bits_path),
            "recon_path": str(recon_path),
            "source_image_path": str(source_image_path),
            "status": "ok",
            "error": "",
        }

        try:
            img, _ = dataset[int(stl_idx)]
            img.save(str(source_image_path))

            tx_result = J.Tx(
                str(source_image_path),
                rx_wav_path=str(rx_wav_path),
                ams22_device_index=args.input_device_idx,
                tx_output_device_index=args.output_device_idx,
                rx_channels=int(args.rx_channels),
                rx_samplerate=int(args.sample_rate),
                center_frequency_hz=float(args.center_freq),
                force_offline_loopback=bool(args.force_offline_loopback),
            )
            tx_result = tx_result if isinstance(tx_result, dict) else {}

            src_tx_wav = Path(tx_result.get("tx_wav_path", str(J.TX_WAV_PATH)))
            src_tx_bits = Path(tx_result.get("tx_bitstream_path", str(J.TX_BITSTREAM_PATH)))
            if src_tx_wav.is_file():
                shutil.copy2(str(src_tx_wav), str(tx_wav_dst))
            if src_tx_bits.is_file():
                shutil.copy2(str(src_tx_bits), str(tx_bits_dst))

            J.Rx(
                str(rx_bits_path),
                save_img_path=str(recon_path),
                rx_wav_path=str(rx_wav_path),
                center_frequency_hz=float(args.center_freq),
            )

            phy = dict(J.LAST_PHY_STATS) if isinstance(J.LAST_PHY_STATS, dict) else {}
            wav_diag = _probe_wav(rx_wav_path)
            ber = _compute_ber(tx_bits_dst, rx_bits_path)
            payload_bits = _payload_bits(tx_bits_dst)

            record.update(
                {
                    "rx_rms": _safe_float(
                        wav_diag.get("rx_rms", phy.get("rx_passband_rms"))
                    ),
                    "sync_peak": _safe_float(phy.get("sync_peak")),
                    "estimated_snr": _extract_estimated_snr(phy),
                    "ber": _safe_float(ber if ber is not None else phy.get("ber")),
                    "payload_bits": payload_bits,
                    "sample_rate": int(wav_diag.get("rx_samplerate", args.sample_rate)),
                    "transmission_mode": tx_result.get("transmission_mode", ""),
                }
            )
        except Exception as e:
            record["status"] = "error"
            record["error"] = str(e)

        _append_jsonl(jsonl_path, record)
        _append_csv(csv_path, record, fieldnames=csv_fields)

        print(
            f"[{i + 1:04d}/{args.num_samples}] {sample_name} "
            f"status={record['status']} ber={record['ber']} rx_rms={record['rx_rms']}"
        )

    print(f"[DONE] JSONL: {jsonl_path}")
    print(f"[DONE] CSV:   {csv_path}")


if __name__ == "__main__":
    main()
