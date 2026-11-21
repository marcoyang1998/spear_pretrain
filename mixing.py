# Copyright    2021-2023  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Zengwei Yao)
#
# Copyright    2024 University of Cambridge      (authors: Xiaoyu Yang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Reference waveform and target-mixing helpers; no external noise dataset."""
import math
import random
import numpy as np
import torch
from lhotse.cut import Cut, MixedCut
from lhotse.cut.set import mix

def mix_with_offset(
    reference_cut: Cut,
    mixed_in_cut: Cut,
    snr: float = 10.0,
    drop_mixed_in_supervision: bool = True,
    *,
    # 仅对“语音重叠”模式——如果是噪声注入，建议另行分支处理
    min_overlap_ratio: float = 0.20,   # 下限
    max_overlap_ratio: float = 0.50,   # 上限
    epsilon: float = 0.01,             # tolerance
):
    if drop_mixed_in_supervision and hasattr(mixed_in_cut, "drop_supervisions"):
        mixed_in_cut = mixed_in_cut.drop_supervisions()

    ref_duration = float(reference_cut.duration)
    if ref_duration <= (0.1 + epsilon):
        return reference_cut  # 极短段保护

    # 计算严格 < 50% 的上界
    max_allowed = max(0.0, max_overlap_ratio * ref_duration - epsilon)
    min_allowed = max(0.1, min_overlap_ratio * ref_duration)  # 0.1s 安全下限与原逻辑一致
    if max_allowed < min_allowed:  # 容错：极短段或参数不当
        min_allowed = max(0.05, min_allowed * 0.5)
        max_allowed = max(min_allowed + epsilon, max_allowed)

    mix_duration = random.uniform(min_allowed, max_allowed)

    # 截断被混入段以满足目标 mix_duration
    mixed_in_duration = float(mixed_in_cut.duration)
    if mixed_in_duration > mix_duration:
        # 在可行范围内随机起点后截断
        slack = max(0.0, mixed_in_duration - mix_duration - epsilon)
        truncate_start = random.uniform(0.0, slack) if slack > 0 else 0.0
        mixed_in_cut = mixed_in_cut.truncate(offset=truncate_start, duration=mix_duration)

    actual_mix_duration = min(float(mixed_in_cut.duration), mix_duration)

    # 将被混入段完全放入参考段内部，保证真实 overlap = actual_mix_duration
    hi = max(0.0, ref_duration - actual_mix_duration - epsilon)
    offset = random.uniform(0.0, hi) if hi > 0 else 0.0

    mixed_cut = mix(
        reference_cut=reference_cut,
        mixed_in_cut=mixed_in_cut,
        offset=offset,
        snr=snr,
        preserve_id="left",
    )
    return mixed_cut

def mix_with_delay_max(
    reference_cut: Cut,
    mixed_in_cut: Cut,
    snr: float = 10.0,
    delay: float = 10.0,
    drop_mixed_in_supervision: bool = True,
):
    """
    Mix two cuts with a random delay offset. The mixed_in_cut is not truncated,
    so the resulting cut can be longer than the reference_cut.
    
    Args:
        reference_cut: The reference cut to mix into
        mixed_in_cut: The cut to mix in with a delay
        snr: Signal-to-noise ratio for mixing
        delay: Maximum delay in seconds (actual delay will be sampled from [0, min(delay, reference_cut.duration)])
        drop_mixed_in_supervision: Whether to drop supervisions from the mixed_in_cut
    
    Returns:
        A mixed cut that may be longer than the reference_cut
    """
    if drop_mixed_in_supervision and hasattr(mixed_in_cut, "drop_supervisions"):
        mixed_in_cut = mixed_in_cut.drop_supervisions()
    
    ref_duration = float(reference_cut.duration)
    
    # Sample delay from [0, min(delay, ref_duration)]
    max_delay = min(delay, ref_duration)
    offset = random.uniform(0.0, max_delay)
    
    # Mix without truncating mixed_in_cut
    mixed_cut = mix(
        reference_cut=reference_cut,
        mixed_in_cut=mixed_in_cut,
        offset=offset,
        snr=snr,
        preserve_id="left",
    )
    return mixed_cut

def compute_prob_from_energies(gain: float, energy_main: float, energy_mixin: float) -> float:
    """
    Computes the probability of selecting the mix-in token based on the 
    actual energy contribution of the mix-in signal to the total mixture.
    
    Args:
        gain: The linear gain applied to the mix-in signal (amplitude).
        energy_main: The energy (mean squared) of the main signal (A).
        energy_mixin: The energy (mean squared) of the mix-in signal (B) BEFORE gain.
        
    Returns:
        float: Probability between 0.0 and 1.0 representing the mix-in's dominance.
    """
    # 1. Calculate the energy of the mix-in signal AFTER the gain is applied.
    # Since Energy ~ Amplitude^2, we square the gain.
    energy_mixin_effective = (gain ** 2) * energy_mixin
    
    # 2. Compute total energy (assuming signals are uncorrelated).
    energy_total = energy_main + energy_mixin_effective
    
    # 3. Compute the ratio of the mix-in energy to the total energy.
    # Add epsilon to prevent division by zero.
    prob = energy_mixin_effective / (energy_total + 1e-8)
    
    return float(prob)

def audio_energy(audio: np.ndarray):
    # return the average energy of the audio
    return float(np.average(audio**2))

def _load_mixed_cut_single2(cut: MixedCut):
    # ---------------------------------------------------------
    # 1. Load Components & Safety Checks
    # ---------------------------------------------------------
    sample_rate = cut.sampling_rate
    track_sig = cut.tracks[0]
    track_noise = cut.tracks[1]
    
    # 强制转换为 float32 以避免 int16 溢出或精度丢失
    orig_audio = track_sig.cut.load_audio()
    mix_in_audio = track_noise.cut.load_audio()
    
    # get the power and compute the gain
    power_sig = audio_energy(orig_audio)
    power_noise = audio_energy(mix_in_audio)
    target_noise_power = power_sig * (10.0 ** (-track_noise.snr / 10))
    
    if power_noise > 1e-12:
        # Gain = sqrt(Target / Source)
        gain = math.sqrt(target_noise_power / power_noise)
    else:
        # 噪音文件是静音
        gain = 0.0
        
    # compute the replacement probability
    replacement_prob = compute_prob_from_energies(gain, power_sig, power_noise)
    
    # compute the resulting audio
    result_audio = cut.load_audio()
    
    return result_audio, gain, replacement_prob

def _mix_tokens_single(A: torch.Tensor, B: torch.Tensor, p: float) -> torch.Tensor:
    """
    从 A 中随机选出 p% 的位置，用 B 中对应位置的值替换。
    
    参数:
        A (Tensor): 原始张量，形状为 (T, C)
        B (Tensor): 替换来源张量，形状必须与 A 相同
        p (float): 替换比例，范围为 0~1

    返回:
        Tensor: 替换后的新张量
    """
    assert A.shape == B.shape, "A and B must have the same shape"
    assert 0 <= p <= 1, "p must be between 0 and 1"
    
    # 创建一个与 A 相同形状的 mask，表示哪些位置需要替换
    mask = torch.rand_like(A, dtype=torch.float32) < p

    # 创建新的张量：如果 mask 为 True，就用 B 的值，否则用 A 的值
    return torch.where(mask, B, A)
