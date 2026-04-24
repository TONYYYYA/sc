"""
PracticalBellhopChannel - 完全基于 ofdm_python.matlab_port 的实现
保留原实现的所有关键特性：
- MultiBitSTEQuantizer 非均匀量化
- RobustVectorQuantizer 支持
- UEP (Unequal Error Protection)
- Gray 编码/解码
"""
import os
import sys
import glob
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

# Add project root to path to import bellhop_real
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import bellhop reader
try:
    from bellhop_real.simulate_ofdm_ber_strict import read_arrivals_asc_strict
except ImportError:
    print("Error: Could not import bellhop_real.simulate_ofdm_ber_strict")
    sys.exit(1)

from pic_compress.model import CompressAIJSCCModel
from pic_compress.utils import compute_psnr
from ofdm_python.matlab_port import run_matlab_style_ofdm_bridge_from_bits, MATLABOFDMConfig

# 导入LSQ量化器
try:
    from pic_compress.quantizers import LSQQuantizer, LSQWrapper
except ImportError:
    from quantizers import LSQQuantizer, LSQWrapper


def _build_nonuniform_levels(bits, clip_val, device=None, dtype=None, mode="tanh", compand_mu=6.0):
    n_levels = 1 << int(bits)
    base = torch.linspace(-1.0, 1.0, steps=n_levels, device=device, dtype=dtype or torch.float32)
    mode = str(mode).lower()
    if mode == "uniform":
        levels = base
    elif mode == "mulaw":
        mu = float(max(1e-3, compand_mu))
        levels = torch.sign(base) * torch.log1p(mu * torch.abs(base)) / math.log1p(mu)
    else:
        levels = torch.tanh(1.5 * base) / math.tanh(1.5)
    return levels * float(clip_val)


def gray_encode_indices(indices, bits):
    bits = int(bits)
    gray = torch.bitwise_xor(indices, torch.bitwise_right_shift(indices, 1))
    out = []
    for shift in range(bits - 1, -1, -1):
        out.append(torch.bitwise_and(torch.bitwise_right_shift(gray, shift), 1))
    return torch.stack(out, dim=-1).to(torch.float32)


def gray_decode_soft_bits(bit_soft_values, bits=2, clip_val=1.0, llr_scale=2.0, quant_mode="tanh", compand_mu=6.0):
    bits = int(bits)
    levels = _build_nonuniform_levels(bits, clip_val, device=bit_soft_values.device, dtype=bit_soft_values.dtype, mode=quant_mode, compand_mu=compand_mu)
    probs = torch.sigmoid(bit_soft_values * float(llr_scale))
    out = torch.zeros(bit_soft_values.shape[:-1], device=bit_soft_values.device, dtype=bit_soft_values.dtype)
    for idx in range(1 << bits):
        gray = idx ^ (idx >> 1)
        prob = torch.ones_like(out)
        for bit_pos in range(bits):
            bit = (gray >> (bits - 1 - bit_pos)) & 1
            p = probs[..., bit_pos]
            prob = prob * (p if bit == 1 else (1.0 - p))
        out = out + prob * levels[idx]
    return out


def gray_decode_soft_vectors(bit_soft_values, codebook, bits_per_group, llr_scale=2.0):
    probs = torch.sigmoid(bit_soft_values * float(llr_scale))
    bsz = probs.shape[0]
    num_groups = probs.shape[1] // bits_per_group
    probs_reshaped = probs[:, :num_groups * bits_per_group].view(bsz, num_groups, bits_per_group)
    out = []
    for g in range(num_groups):
        group_probs = probs_reshaped[:, g, :]
        group_llr = torch.log(group_probs + 1e-10) - torch.log(1.0 - group_probs + 1e-10)
        # codebook 的维度是 (num_embeddings, embedding_dim)，但 group_llr 的维度是 bits_per_group
        # 需要将 codebook 转换为与 group_llr 相同的维度
        if codebook.shape[1] != bits_per_group:
            # 如果维度不匹配，使用简单的最近邻查找（基于索引）
            # 计算每个比特的概率，然后选择最可能的索引
            indices = []
            for i in range(codebook.shape[0]):
                # 将 codebook 索引转换为比特模式
                idx_bits = [(i >> j) & 1 for j in range(bits_per_group)]
                idx_probs = torch.tensor(idx_bits, dtype=group_probs.dtype, device=group_probs.device)
                # 计算与当前概率的匹配度
                match = -torch.sum((group_probs - idx_probs) ** 2, dim=-1)
                indices.append(match)
            indices = torch.stack(indices, dim=-1)  # (bsz, num_embeddings)
            closest_idx = torch.argmax(indices, dim=-1)
        else:
            distances = torch.cdist(group_llr.unsqueeze(1), codebook.unsqueeze(0), p=2).squeeze(1)
            closest_idx = torch.argmin(distances, dim=-1)
        out.append(codebook[closest_idx])
    return torch.cat(out, dim=-1)


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


def apply_symbol_repeat_range(symbol_bits, protected_start, protected_len, repeat_factor=2):
    num_symbols = int(symbol_bits.shape[-2])
    protected_start = max(0, min(num_symbols, int(protected_start)))
    protected_len = max(0, min(num_symbols - protected_start, int(protected_len)))
    repeat_factor = int(repeat_factor)
    if repeat_factor <= 1 or protected_len <= 0:
        return symbol_bits, protected_start, protected_len
    head = symbol_bits[..., :protected_start, :]
    protected = symbol_bits[..., protected_start:protected_start + protected_len, :]
    tail = symbol_bits[..., protected_start + protected_len:, :]
    encoded = torch.cat([head] + [protected for _ in range(repeat_factor)] + [tail], dim=-2)
    return encoded, protected_start, protected_len


def merge_symbol_repeat_range(symbol_soft, original_symbols, protected_start, protected_len, repeat_factor=2):
    original_symbols = int(original_symbols)
    protected_start = max(0, min(original_symbols, int(protected_start)))
    protected_len = max(0, min(original_symbols - protected_start, int(protected_len)))
    repeat_factor = int(repeat_factor)
    if repeat_factor <= 1 or protected_len <= 0:
        return symbol_soft[..., :original_symbols, :]
    head = symbol_soft[..., :protected_start, :]
    rep_start = protected_start
    rep_end = rep_start + protected_len * repeat_factor
    rep = symbol_soft[..., rep_start:rep_end, :]
    rep_chunks = rep.reshape(*symbol_soft.shape[:-2], repeat_factor, protected_len, symbol_soft.shape[-1])
    merged = rep_chunks.mean(dim=-3)
    tail_start = rep_end
    tail_len = max(0, original_symbols - protected_start - protected_len)
    tail = symbol_soft[..., tail_start:tail_start + tail_len, :]
    return torch.cat([head, merged, tail], dim=-2)


def group_bits_to_qpsk_symbols(tx_bits):
    total_bits = int(tx_bits.shape[-1])
    if total_bits % 2 != 0:
        pad = torch.zeros(*tx_bits.shape[:-1], 1, device=tx_bits.device, dtype=tx_bits.dtype)
        tx_bits = torch.cat([tx_bits, pad], dim=-1)
        total_bits += 1
    tx_symbol_bits = tx_bits.reshape(*tx_bits.shape[:-1], total_bits // 2, 2)
    return tx_symbol_bits, total_bits, total_bits // 2


def ungroup_qpsk_symbol_bits(y_symbol_soft, total_payload_bits, original_shape):
    B = original_shape[0]
    flat = y_symbol_soft.reshape(B, -1)
    if flat.shape[-1] > total_payload_bits:
        flat = flat[..., :total_payload_bits]
    elif flat.shape[-1] < total_payload_bits:
        pad_len = total_payload_bits - flat.shape[-1]
        pad = torch.zeros(B, pad_len, device=flat.device, dtype=flat.dtype)
        flat = torch.cat([flat, pad], dim=-1)
    return flat.reshape(original_shape)


def insert_proxy_pilots(tx_syms, frame_size=64, pilot_stride=16, pilot_value=1.0):
    frame_size = int(max(8, frame_size))
    pilot_stride = int(max(2, min(frame_size, pilot_stride)))
    bsz, data_len = tx_syms.shape
    pilot_mask = torch.zeros(frame_size, dtype=torch.bool, device=tx_syms.device)
    pilot_mask[::pilot_stride] = True
    data_mask = ~pilot_mask
    data_indices = torch.nonzero(data_mask, as_tuple=False).flatten()
    payload_per_frame = int(data_indices.numel())
    num_frames = int(math.ceil(float(data_len) / float(payload_per_frame)))
    framed = tx_syms.new_zeros((bsz, num_frames, frame_size))
    framed[..., pilot_mask] = complex(float(pilot_value), 0.0)
    cursor = 0
    for frame_idx in range(num_frames):
        take = min(payload_per_frame, data_len - cursor)
        if take > 0:
            framed[:, frame_idx, data_indices[:take]] = tx_syms[:, cursor:cursor + take]
            cursor += take
    return framed, data_mask, pilot_mask, data_len


def extract_proxy_payload(y_eq, data_mask, data_len):
    bsz = y_eq.shape[0]
    payload_per_frame = int(data_mask.sum().item())
    out = []
    cursor = 0
    for frame_idx in range(y_eq.shape[1]):
        frame_payload = y_eq[:, frame_idx, data_mask]
        take = min(payload_per_frame, data_len - cursor)
        if take > 0:
            out.append(frame_payload[:, :take])
            cursor += take
    return torch.cat(out, dim=-1) if out else y_eq.new_zeros((bsz, 0))


def estimate_channel_from_pilots(y_syms, pilot_mask, pilot_value):
    if y_syms.dim() == 2:
        pilot_obs = y_syms[:, pilot_mask]
        h_pilots = pilot_obs / complex(float(pilot_value), 0.0)
        real = F.interpolate(h_pilots.real.unsqueeze(1), size=y_syms.shape[1], mode="linear", align_corners=True).squeeze(1)
        imag = F.interpolate(h_pilots.imag.unsqueeze(1), size=y_syms.shape[1], mode="linear", align_corners=True).squeeze(1)
        return torch.complex(real, imag)
    if y_syms.dim() != 3:
        raise ValueError(f"Unsupported pilot estimation shape: {tuple(y_syms.shape)}")
    bsz, num_frames, frame_size = y_syms.shape
    pilot_obs = y_syms[..., pilot_mask]
    h_pilots = pilot_obs / complex(float(pilot_value), 0.0)
    interp_in = h_pilots.reshape(bsz * num_frames, 1, -1)
    real = F.interpolate(interp_in.real, size=frame_size, mode="linear", align_corners=True).reshape(bsz, num_frames, frame_size)
    imag = F.interpolate(interp_in.imag, size=frame_size, mode="linear", align_corners=True).reshape(bsz, num_frames, frame_size)
    return torch.complex(real, imag)


def pilot_latents_to_qpsk_symbol_offset(latents, pilot_value=1.0, clip=3.0):
    latents = torch.clamp(latents, -clip, clip)
    angle = torch.atan2(latents[..., 1], latents[..., 0])
    mag = torch.sqrt(latents[..., 0]**2 + latents[..., 1]**2).clamp_min(1e-6)
    mag_norm = torch.clamp(mag / mag.mean(dim=-1, keepdim=True).clamp_min(1e-6), 0.5, 2.0)
    phase_offset = angle / math.pi
    amplitude_scale = mag_norm
    return phase_offset, amplitude_scale


class MultiBitSTEQuantizer(nn.Module):
    def __init__(self, bits=2, clip_val=1.0, mode="tanh", compand_mu=6.0):
        super().__init__()
        self.bits = int(max(1, bits))
        self.clip_val = float(clip_val)
        self.mode = str(mode).lower()
        self.compand_mu = float(max(1e-3, compand_mu))
        self.num_levels = 1 << self.bits
        self.register_buffer("levels", _build_nonuniform_levels(self.bits, self.clip_val, mode=self.mode, compand_mu=self.compand_mu))

    def forward(self, x):
        return self.quantize(x)

    def quantize(self, x):
        x_clipped = torch.clamp(x, -self.clip_val, self.clip_val)
        # 确保 levels 在与 x 相同的设备上
        levels = self.levels.to(x.device)
        distances = torch.abs(x_clipped.unsqueeze(-1) - levels.view(1, 1, -1))
        indices = torch.argmin(distances, dim=-1)
        quantized = levels[indices]
        if self.training:
            return x_clipped + (quantized - x_clipped).detach()
        return quantized

    def quantize_with_indices(self, x):
        x_clipped = torch.clamp(x, -self.clip_val, self.clip_val)
        # 确保 levels 在与 x 相同的设备上
        levels = self.levels.to(x.device)
        distances = torch.abs(x_clipped.unsqueeze(-1) - levels.view(1, 1, -1))
        indices = torch.argmin(distances, dim=-1)
        quantized = levels[indices]
        if self.training:
            return x_clipped + (quantized - x_clipped).detach(), indices
        return quantized, indices


class RobustVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=256, embedding_dim=2, commitment_beta=0.25, noise_std=0.06):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.commitment_beta = float(commitment_beta)
        self.noise_std = float(noise_std)
        self.codebook = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)
        self.bits_per_group = int(math.log2(self.num_embeddings))
        self.group_dim = self.embedding_dim
        self.last_vq_loss = None

    def forward(self, z):
        z_q, vq_loss = self.quantize(z)
        self.last_vq_loss = vq_loss
        return z_q

    def quantize(self, z):
        z_flat = z.reshape(-1, self.embedding_dim)
        distances = torch.cdist(z_flat, self.codebook.weight, p=2)
        indices = torch.argmin(distances, dim=1)
        z_q_flat = self.codebook(indices)
        z_q = z_q_flat.reshape(z.shape)
        commitment_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach())
        vq_loss = codebook_loss + self.commitment_beta * commitment_loss
        if self.training:
            z_q = z + (z_q - z).detach()
            if self.noise_std > 0:
                z_q = z_q + torch.randn_like(z_q) * self.noise_std
        return z_q, vq_loss

    def quantize_with_indices(self, z):
        z_flat = z.reshape(-1, self.embedding_dim)
        distances = torch.cdist(z_flat, self.codebook.weight, p=2)
        indices = torch.argmin(distances, dim=1)
        z_q_flat = self.codebook(indices)
        z_q = z_q_flat.reshape(z.shape)
        commitment_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach())
        vq_loss = codebook_loss + self.commitment_beta * commitment_loss
        self.last_vq_loss = vq_loss
        if self.training:
            z_q = z + (z_q - z).detach()
        return z_q, indices.reshape(z.shape[:-1])


class PracticalBellhopChannel(nn.Module):
    """
    完全基于 ofdm_python.matlab_port 的 PracticalBellhopChannel
    保留原实现的所有关键特性
    优化：支持预计算 BER，避免每次 forward 都进行昂贵的 OFDM 处理
    """
    def __init__(
        self,
        arr_dir,
        fs=144000,
        max_delay=0.02,
        max_paths=32,
        snr_jitter_db=1.5,
        hold_frames=1,
        sample_mode="random",
        cache_arrivals=True,
        max_cache_files=512,
        quant_bits=2,
        quant_clip_val=1.0,
        quant_mode="tanh",
        quant_compand_mu=6.0,
        quant_bits_min=None,
        quant_bits_max=None,
        use_robust_vq=False,
        vq_group_dim=2,
        vq_commitment_beta=0.25,
        vq_noise_std=0.06,
        vq_max_bits_per_group=14,
        llr_scale=2.0,
        uep_protect_ratio=0.5,
        uep_repeat_factor=2,
        proxy_frame_size=64,
        pilot_stride=16,
        pilot_value=1.0,
        use_precomputed_ber=True,
        precomputed_ber_path=None,
        allowed_buckets=None,
        **kwargs
    ):
        super().__init__()
        self.arr_dir = arr_dir
        self.fs = fs
        self.max_delay = max_delay
        self.max_paths = max_paths
        self.snr_jitter_db = snr_jitter_db
        self.hold_frames = int(max(1, hold_frames))
        self.sample_mode = str(sample_mode)
        self.cache_arrivals = bool(cache_arrivals)
        self.max_cache_files = int(max(1, max_cache_files))
        self.quant_bits = int(max(1, quant_bits))
        self.quant_clip_val = float(quant_clip_val)
        self.quant_mode = str(quant_mode).lower()
        self.quant_compand_mu = float(max(1e-3, quant_compand_mu))
        self.quant_bits_min = None if quant_bits_min is None else int(max(1, quant_bits_min))
        self.quant_bits_max = None if quant_bits_max is None else int(max(self.quant_bits, quant_bits_max))
        self.use_robust_vq = bool(use_robust_vq)
        self.vq_group_dim = int(vq_group_dim)
        self.vq_commitment_beta = float(vq_commitment_beta)
        self.vq_noise_std = float(vq_noise_std)
        self.vq_max_bits_per_group = int(vq_max_bits_per_group)
        self.llr_scale = float(llr_scale)
        self.uep_protect_ratio = float(uep_protect_ratio)
        self.uep_repeat_factor = int(uep_repeat_factor)
        self.proxy_frame_size = int(proxy_frame_size)
        self.pilot_stride = int(pilot_stride)
        self.pilot_value = float(pilot_value)
        self.use_precomputed_ber = bool(use_precomputed_ber)
        self.allowed_buckets = set(allowed_buckets) if allowed_buckets else None
        
        # OFDM 配置
        self.ofdm_cfg = MATLABOFDMConfig(fs=fs, snr_db=10.0)
        
        # Robust VQ
        self.robust_vq = None
        if self.use_robust_vq:
            bits_per_group = min(self.vq_max_bits_per_group, max(1, self.quant_bits))
            num_embeddings = 1 << bits_per_group
            self.robust_vq = RobustVectorQuantizer(
                num_embeddings=num_embeddings,
                embedding_dim=self.vq_group_dim,
                commitment_beta=self.vq_commitment_beta,
                noise_std=self.vq_noise_std,
            )
        
        self._arr_cache = {}
        self._arr_cursor = 0
        self._hold_left = 0
        self._hold_h = None
        self.last_vq_loss = None
        self._current_arr_name = None
        
        # 加载预计算的 BER
        self.precomputed_ber = {}
        self.precomputed_buckets = {}
        if self.use_precomputed_ber:
            self._load_precomputed_ber(precomputed_ber_path)
        
        # 如果只使用预计算 BER 的文件，则过滤 arr_files
        if self.use_precomputed_ber and self.precomputed_ber:
            all_arr_files = glob.glob(os.path.join(arr_dir, "**/*.arr"), recursive=True)
            # 只保留有预计算 BER 的文件
            self.arr_files = [f for f in all_arr_files if os.path.basename(f) in self.precomputed_ber]
            # 如果指定了允许的桶，则进一步过滤
            if self.allowed_buckets:
                self.arr_files = [f for f in self.arr_files if self.precomputed_buckets.get(os.path.basename(f)) in self.allowed_buckets]
            if not self.arr_files:
                raise ValueError(f"No ARR files with precomputed BER found in {arr_dir} (allowed_buckets: {self.allowed_buckets})")
            print(f"PracticalBellhopChannel (MATLAB-style): Using {len(self.arr_files)} ARR files with precomputed BER")
            if self.allowed_buckets:
                print(f"  Filtered by buckets: {self.allowed_buckets}")
        else:
            self.arr_files = glob.glob(os.path.join(arr_dir, "**/*.arr"), recursive=True)
            if not self.arr_files:
                raise ValueError(f"No .arr files found in {arr_dir}")
            print(f"PracticalBellhopChannel (MATLAB-style): Found {len(self.arr_files)} real channel files.")

    def _load_precomputed_ber(self, precomputed_ber_path=None):
        """加载预计算的 BER 文件"""
        if precomputed_ber_path is None:
            # 默认路径
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            precomputed_ber_path = os.path.join(project_root, "ofdm_python", "arr_ber_scan.json")
        
        if not os.path.exists(precomputed_ber_path):
            print(f"Warning: Precomputed BER file not found: {precomputed_ber_path}")
            print("  Falling back to full OFDM processing (slower)")
            self.use_precomputed_ber = False
            return
        
        try:
            with open(precomputed_ber_path, 'r') as f:
                scan_data = json.load(f)
            
            rows = scan_data.get('rows', [])
            for row in rows:
                arr_name = row['arr']
                self.precomputed_ber[arr_name] = row['ber']
                self.precomputed_buckets[arr_name] = row['bucket']
            
            print(f"Loaded precomputed BER for {len(self.precomputed_ber)} ARR files")
            print(f"  SNR: {scan_data.get('snr_db')} dB")
            print(f"  Buckets: {scan_data.get('buckets')}")
        except Exception as e:
            print(f"Warning: Failed to load precomputed BER: {e}")
            print("  Falling back to full OFDM processing (slower)")
            self.use_precomputed_ber = False

    def _simulate_errors_with_ber(self, tx_bits, ber):
        """根据 BER 直接模拟错误（GPU 加速）"""
        device = tx_bits.device
        
        # 生成随机错误位置
        error_mask = torch.rand(tx_bits.shape, device=device) < ber
        
        # 翻转错误位置的比特
        rx_bits = tx_bits.clone()
        rx_bits[error_mask] = 1 - rx_bits[error_mask]
        
        return rx_bits

    def _sample_quant_bits(self):
        if not self.training:
            return self.quant_bits
        if self.quant_bits_min is None or self.quant_bits_max is None or self.quant_bits_max <= self.quant_bits_min:
            return self.quant_bits
        return random.randint(self.quant_bits_min, self.quant_bits_max)

    def _pick_arr_path(self):
        if self.sample_mode == "cycle":
            arr_path = self.arr_files[self._arr_cursor % len(self.arr_files)]
            self._arr_cursor = (self._arr_cursor + 1) % len(self.arr_files)
        else:
            arr_path = random.choice(self.arr_files)
        # 保存当前 ARR 文件名用于日志
        self._current_arr_name = os.path.basename(arr_path)
        return arr_path

    def _process_single_ofdm_frame(self, args):
        """处理单个 OFDM 帧（用于并行处理）"""
        tx_chunk, arr_path, snr_db = args
        try:
            result = run_matlab_style_ofdm_bridge_from_bits(tx_chunk, arr_path, snr_db=snr_db)
            return result['rx_bits']
        except Exception as e:
            print(f"Warning: OFDM frame processing failed: {e}")
            return tx_chunk  # 回退

    def _process_with_ofdm(self, tx_bits_np, arr_path, snr_db):
        """使用 ofdm_python.matlab_port 处理比特 - 支持并行分块传输"""
        try:
            # ofdm_python 需要特定数量的比特
            # 使用 5000 比特作为标准帧大小
            ofdm_num_bits = 5000
            total_bits = len(tx_bits_np)
            
            # 如果总比特数小于等于帧大小，直接处理
            if total_bits <= ofdm_num_bits:
                if total_bits < ofdm_num_bits:
                    # 填充到所需长度
                    repeats = (ofdm_num_bits + total_bits - 1) // total_bits
                    tx_bits_padded = np.tile(tx_bits_np, repeats)[:ofdm_num_bits]
                else:
                    tx_bits_padded = tx_bits_np
                
                result = run_matlab_style_ofdm_bridge_from_bits(tx_bits_padded, arr_path, snr_db=snr_db)
                rx_bits = result['rx_bits'][:total_bits]
                return rx_bits
            
            # 分块处理：将大数据块分成多个 5000 比特的帧
            num_full_frames = total_bits // ofdm_num_bits
            remaining_bits = total_bits % ofdm_num_bits
            
            # 准备并行处理的任务列表
            tasks = []
            
            # 处理完整的帧
            for i in range(num_full_frames):
                start_idx = i * ofdm_num_bits
                end_idx = start_idx + ofdm_num_bits
                tx_chunk = tx_bits_np[start_idx:end_idx]
                tasks.append((tx_chunk, arr_path, snr_db))
            
            # 处理剩余的部分（如果有）
            if remaining_bits > 0:
                start_idx = num_full_frames * ofdm_num_bits
                tx_remaining = tx_bits_np[start_idx:]
                
                # 填充到 5000 比特
                repeats = (ofdm_num_bits + remaining_bits - 1) // remaining_bits
                tx_padded = np.tile(tx_remaining, repeats)[:ofdm_num_bits]
                tasks.append((tx_padded, arr_path, snr_db))
            
            # 并行处理所有帧
            max_workers = min(mp.cpu_count(), 8)  # 最多使用8个进程
            rx_bits_list = []
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(self._process_single_ofdm_frame, tasks))
            
            # 收集结果
            for i, rx_bits in enumerate(results):
                if i < num_full_frames:
                    rx_bits_list.append(rx_bits)
                else:
                    # 最后一个可能是填充的，需要截取
                    rx_bits_list.append(rx_bits[:remaining_bits])
            
            # 合并所有接收的比特
            rx_bits = np.concatenate(rx_bits_list)
            return rx_bits
            
        except Exception as e:
            print(f"Warning: OFDM processing failed: {e}")
            return tx_bits_np  # 回退

    def forward(self, x, snr_db=None, mode='ofdm'):
        """
        前向传播 - 保留原实现的所有关键特性
        """
        input_shape = x.shape
        device = x.device
        
        # 展平输入
        if x.dim() >= 3:
            B = x.shape[0]
            x_flat = x.reshape(B, -1)
        elif x.dim() == 2:
            B = x.shape[0]
            x_flat = x
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        
        L = x_flat.shape[1]
        
        # 处理 SNR
        snr_tensor = None
        if snr_db is not None:
            if isinstance(snr_db, (int, float)):
                snr_tensor = torch.full((B, 1), float(snr_db), device=device)
            elif isinstance(snr_db, torch.Tensor):
                snr_tensor = snr_db.to(device)
                if snr_tensor.ndim == 0:
                    snr_tensor = snr_tensor.view(1, 1).expand(B, 1)
                elif snr_tensor.ndim == 1:
                    if snr_tensor.shape[0] == 1:
                        snr_tensor = snr_tensor.view(1, 1).expand(B, 1)
                    else:
                        snr_tensor = snr_tensor.view(-1, 1)
            if snr_tensor is not None and self.training and self.snr_jitter_db > 0:
                snr_tensor = snr_tensor + torch.randn_like(snr_tensor) * self.snr_jitter_db
        
        self.last_vq_loss = None
        active_quant_bits = self._sample_quant_bits()
        
        # 处理每个样本
        outputs = []
        for b in range(B):
            x_b = x_flat[b:b+1]  # (1, L)
            snr_b = snr_tensor[b].item() if snr_tensor is not None else 10.0
            
            if self.use_robust_vq and self.robust_vq is not None:
                # Robust VQ 模式
                G = self.robust_vq.group_dim
                bits_per_group = self.robust_vq.bits_per_group
                pad = (G - (L % G)) % G
                if pad > 0:
                    x_pad = F.pad(x_b, (0, pad), value=0.0)
                else:
                    x_pad = x_b
                num_groups = x_pad.shape[1] // G
                z = x_pad.view(1, num_groups, G)
                x_tx_pad, vq_idx = self.robust_vq.quantize_with_indices(z)
                x_tx_flat = x_tx_pad.view(1, -1)
                x_tx = x_tx_flat[:, :L]
                
                if self.robust_vq.last_vq_loss is not None:
                    self.last_vq_loss = self.robust_vq.last_vq_loss
                
                # VQ 索引转换为比特
                tx_bits = gray_encode_indices(vq_idx, bits_per_group).view(1, -1)
                tx_symbol_bits, total_payload_bits, _ = group_bits_to_qpsk_symbols(tx_bits)
                original_symbols = tx_symbol_bits.shape[1]
                
                # UEP 保护
                total_bits = int(num_groups * bits_per_group)
                base_bits, res_bits = split_quant_bits(active_quant_bits, preferred_base_bits=3, min_res_bits=1)
                base_payload_bits = int(round(total_bits * (float(base_bits) / float(max(1, active_quant_bits)))))
                protected_start = 0
                protected_len = int(math.ceil(float(base_payload_bits) / 2.0))
                
                tx_bits_uep, protected_start, protected_len = apply_symbol_repeat_range(
                    tx_symbol_bits,
                    protected_start=protected_start,
                    protected_len=protected_len,
                    repeat_factor=self.uep_repeat_factor,
                )
                
                # QPSK 调制
                b0 = tx_bits_uep[..., 0]
                b1 = tx_bits_uep[..., 1]
                qpsk_real = b0 * 2.0 - 1.0
                qpsk_imag = 1.0 - b1 * 2.0
                tx_syms = torch.complex(qpsk_real, qpsk_imag) / math.sqrt(2.0)
                
                # 插入导频
                tx_syms, data_mask, pilot_mask, data_len = insert_proxy_pilots(
                    tx_syms,
                    frame_size=self.proxy_frame_size,
                    pilot_stride=self.pilot_stride,
                    pilot_value=self.pilot_value,
                )
                
                # 转换为 numpy 进行 OFDM 处理
                # 将复数符号转换为比特流（用于 ofdm_python）
                tx_syms_np = tx_syms.squeeze(0).cpu().numpy()
                
                # 将 QPSK 符号转换为比特
                tx_bits_list = []
                for sym in tx_syms_np.flatten():
                    real_bit = 1 if sym.real > 0 else 0
                    imag_bit = 0 if sym.imag > 0 else 1  # 注意：imag 是 1 - b1 * 2
                    tx_bits_list.extend([real_bit, imag_bit])
                tx_bits_np = np.array(tx_bits_list, dtype=np.int8)
                
                # 选择 ARR 文件
                arr_path = self._pick_arr_path()
                arr_name = os.path.basename(arr_path)
                
                # 使用完整的 OFDM 处理（分块传输）
                rx_bits_np = self._process_with_ofdm(tx_bits_np, arr_path, snr_b)
                
                # 将接收比特转换回复数符号（展平形式）
                rx_syms_list = []
                for i in range(0, len(rx_bits_np), 2):
                    if i + 1 < len(rx_bits_np):
                        real = 1.0 if rx_bits_np[i] == 1 else -1.0
                        imag = -1.0 if rx_bits_np[i + 1] == 1 else 1.0  # 反向映射
                        rx_syms_list.append(complex(real, imag) / math.sqrt(2.0))
                
                # 提取数据符号（跳过导频位置）
                # data_mask 指示哪些位置是数据（非导频）
                rx_data_syms = []
                data_indices = torch.nonzero(data_mask, as_tuple=False).flatten()
                for frame_idx in range(tx_syms.shape[1]):
                    for data_idx in data_indices:
                        idx = frame_idx * len(data_mask) + data_idx.item()
                        if idx < len(rx_syms_list):
                            rx_data_syms.append(rx_syms_list[idx])
                
                # 转换为张量
                if len(rx_data_syms) > 0:
                    y_data = torch.tensor(rx_data_syms[:data_len], device=device, dtype=torch.complex64).unsqueeze(0)
                else:
                    y_data = torch.zeros((1, data_len), device=device, dtype=torch.complex64)
                
                # 软解调
                y_symbol_soft = torch.stack([y_data.real, -y_data.imag], dim=-1) * math.sqrt(2.0)
                y_symbol_soft = merge_symbol_repeat_range(
                    y_symbol_soft,
                    original_symbols=original_symbols,
                    protected_start=protected_start,
                    protected_len=protected_len,
                    repeat_factor=self.uep_repeat_factor,
                )
                
                # 解组比特
                y_soft = ungroup_qpsk_symbol_bits(y_symbol_soft, total_payload_bits, tx_bits.shape)
                
                # VQ 解码
                vec_hat = gray_decode_soft_vectors(y_soft, self.robust_vq.codebook.weight, bits_per_group, llr_scale=self.llr_scale)
                y_out = vec_hat[:, :L]
                
            else:
                # 标准量化模式 - 支持 STE 或 LSQ
                base_bits, res_bits = split_quant_bits(active_quant_bits, preferred_base_bits=3, min_res_bits=1)
                
                # 基础量化 - 根据 quant_mode 选择量化器
                if self.quant_mode == "lsq":
                    quantizer_base = LSQQuantizer(bits=base_bits, per_channel=True)
                else:
                    quantizer_base = MultiBitSTEQuantizer(bits=base_bits, clip_val=self.quant_clip_val, mode=self.quant_mode, compand_mu=self.quant_compand_mu)
                base_q, base_idx = quantizer_base.quantize_with_indices(x_b)
                
                # 残差量化（如果有）
                if res_bits > 0:
                    res_scale = 1.0 / (1 << base_bits)
                    residual = (x_b - base_q) / res_scale
                    if self.quant_mode == "lsq":
                        quantizer_res = LSQQuantizer(bits=res_bits, per_channel=True)
                    else:
                        quantizer_res = MultiBitSTEQuantizer(bits=res_bits, clip_val=self.quant_clip_val, mode=self.quant_mode, compand_mu=self.quant_compand_mu)
                    res_q, res_idx = quantizer_res.quantize_with_indices(residual)
                else:
                    res_q = torch.zeros_like(x_b)
                    res_idx = torch.zeros_like(base_idx)
                    res_scale = 1.0
                
                # 转换为比特
                tx_bits_base = gray_encode_indices(base_idx, base_bits).view(1, -1)
                tx_bits_res = gray_encode_indices(res_idx, res_bits).view(1, -1) if res_bits > 0 else torch.zeros((1, 0), device=device)
                tx_bits = torch.cat([tx_bits_base, tx_bits_res], dim=-1)
                
                # 分组为 QPSK 符号
                tx_symbol_bits, total_payload_bits, _ = group_bits_to_qpsk_symbols(tx_bits)
                original_symbols = tx_symbol_bits.shape[1]
                
                # UEP 保护
                base_payload_bits = base_idx.numel() * base_bits
                protected_start = 0
                protected_len = int(math.ceil(float(base_payload_bits) / 2.0))
                
                tx_bits_uep, protected_start, protected_len = apply_symbol_repeat_range(
                    tx_symbol_bits,
                    protected_start=protected_start,
                    protected_len=protected_len,
                    repeat_factor=self.uep_repeat_factor,
                )
                
                # QPSK 调制
                b0 = tx_bits_uep[..., 0]
                b1 = tx_bits_uep[..., 1]
                qpsk_real = b0 * 2.0 - 1.0
                qpsk_imag = 1.0 - b1 * 2.0
                tx_syms = torch.complex(qpsk_real, qpsk_imag) / math.sqrt(2.0)
                
                # 插入导频
                tx_syms, data_mask, pilot_mask, data_len = insert_proxy_pilots(
                    tx_syms,
                    frame_size=self.proxy_frame_size,
                    pilot_stride=self.pilot_stride,
                    pilot_value=self.pilot_value,
                )
                
                # 转换为 numpy 进行 OFDM 处理
                tx_syms_np = tx_syms.squeeze(0).cpu().numpy()
                
                # 将 QPSK 符号转换为比特
                tx_bits_list = []
                for sym in tx_syms_np.flatten():
                    real_bit = 1 if sym.real > 0 else 0
                    imag_bit = 0 if sym.imag > 0 else 1
                    tx_bits_list.extend([real_bit, imag_bit])
                tx_bits_np = np.array(tx_bits_list, dtype=np.int8)
                
                # 选择 ARR 文件
                arr_path = self._pick_arr_path()
                arr_name = os.path.basename(arr_path)
                
                # 检查是否可以使用预计算的 BER
                # 使用完整的 OFDM 处理（分块传输）
                rx_bits_np = self._process_with_ofdm(tx_bits_np, arr_path, snr_b)
                
                # 将接收比特转换回复数符号（展平形式）
                rx_syms_list = []
                for i in range(0, len(rx_bits_np), 2):
                    if i + 1 < len(rx_bits_np):
                        real = 1.0 if rx_bits_np[i] == 1 else -1.0
                        imag = -1.0 if rx_bits_np[i + 1] == 1 else 1.0
                        rx_syms_list.append(complex(real, imag) / math.sqrt(2.0))
                
                # 提取数据符号（跳过导频位置）
                rx_data_syms = []
                data_indices = torch.nonzero(data_mask, as_tuple=False).flatten()
                for frame_idx in range(tx_syms.shape[1]):
                    for data_idx in data_indices:
                        idx = frame_idx * len(data_mask) + data_idx.item()
                        if idx < len(rx_syms_list):
                            rx_data_syms.append(rx_syms_list[idx])
                
                # 转换为张量
                if len(rx_data_syms) > 0:
                    y_data = torch.tensor(rx_data_syms[:data_len], device=device, dtype=torch.complex64).unsqueeze(0)
                else:
                    y_data = torch.zeros((1, data_len), device=device, dtype=torch.complex64)
                
                # 软解调
                y_symbol_soft = torch.stack([y_data.real, -y_data.imag], dim=-1) * math.sqrt(2.0)
                y_symbol_soft = merge_symbol_repeat_range(
                    y_symbol_soft,
                    original_symbols=original_symbols,
                    protected_start=protected_start,
                    protected_len=protected_len,
                    repeat_factor=self.uep_repeat_factor,
                )
                
                # 解组比特
                y_soft = ungroup_qpsk_symbol_bits(y_symbol_soft, total_payload_bits, tx_bits.shape)
                
                # 分离基础和残差比特
                base_soft = y_soft[..., :base_bits * base_idx.numel()].view(1, -1, base_bits)
                base_hat = gray_decode_soft_bits(base_soft, bits=base_bits, clip_val=self.quant_clip_val, llr_scale=self.llr_scale, quant_mode=self.quant_mode, compand_mu=self.quant_compand_mu)
                
                if res_bits > 0:
                    res_soft = y_soft[..., base_bits * base_idx.numel():].view(1, -1, res_bits)
                    res_scaled_hat = gray_decode_soft_bits(res_soft, bits=res_bits, clip_val=self.quant_clip_val, llr_scale=self.llr_scale, quant_mode=self.quant_mode, compand_mu=self.quant_compand_mu)
                    y_out = base_hat + res_scaled_hat / (1 << base_bits)
                else:
                    y_out = base_hat
            
            y_out = torch.nan_to_num(y_out, nan=0.0, posinf=0.0, neginf=0.0)
            outputs.append(y_out)
        
        # 堆叠输出
        y_out = torch.cat(outputs, dim=0)
        
        # 恢复原始形状
        return y_out.reshape(input_shape)


def validate_real(model_path, data_path, arr_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Model
    model = CompressAIJSCCModel(N=128, M=192, jscc_hidden=128, channel_out=16).to(device)
    
    # Load Weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"Model loaded from {model_path}")
    
    # Replace Channel with PracticalBellhopChannel
    real_channel = PracticalBellhopChannel(arr_dir, fs=144000, quant_bits=2).to(device)
    model.channel = real_channel
    print("Replaced channel with PracticalBellhopChannel")
    
    model.eval()
    
    # Data
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.STL10(root=data_path, split='test', download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"Starting validation on {len(dataset)} images...")
    
    total_psnr = 0
    count = 0
    
    with torch.no_grad():
        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            x_hat = model(imgs, snr_db=10.0)
            psnr = compute_psnr(imgs, x_hat).item()
            total_psnr += psnr
            count += 1
            
            if i % 10 == 0:
                print(f"Batch {i}: PSNR = {psnr:.2f} dB")
                
            if i >= 50:
                break
                
    avg_psnr = total_psnr / count
    print(f"\nFinal Result on Real Bellhop Channel:")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    
    return avg_psnr


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--arr_dir", type=str, required=True)
    parser.add_argument("--snr", type=float, default=10.0)
    args = parser.parse_args()
    
    print("Testing PracticalBellhopChannel (MATLAB-style)...")
    
    channel = PracticalBellhopChannel(
        arr_dir=args.arr_dir,
        quant_bits=2,
        snr_jitter_db=0.0,
    )
    
    B, L = 2, 100
    x = torch.randn(B, L) * 0.5
    
    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    
    y = channel(x, snr_db=args.snr)
    
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
    
    mse = torch.mean((x - y) ** 2).item()
    print(f"MSE: {mse:.6f}")
    
    print("\nTest passed!")
