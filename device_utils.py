"""
设备工具: 自动检测 NPU/CUDA/CPU, 提供统一接口

支持: 华为昇腾 NPU (torch_npu) / NVIDIA GPU (CUDA) / CPU

用法:
  from device_utils import DEVICE, device_type, autocast, get_scaler, synchronize
  model.to(DEVICE)
  pin_memory = device_supports_pin_memory()
"""
import torch
from contextlib import nullcontext

try:
    import torch_npu
    _HAS_NPU = True
except ImportError:
    torch_npu = None
    _HAS_NPU = False


def get_device():
    """自动检测可用设备: NPU > CUDA > CPU"""
    if _HAS_NPU and torch_npu.npu.is_available():
        d = torch.device("npu:0")
        # 禁用 NPU 内部格式 (NC1HWC0 / NZ), 避免:
        #   1. 随机初始化触发的内部格式告警
        #   2. AdaptiveAvgPool2d / Flatten 等算子反向传播维度不匹配
        #   3. view / reshape 在非标准 strides 上静默失败
        # 性能影响轻微 (标准 NCHW 格式仍可利用 AICore 加速)。
        try:
            torch_npu.npu.config.allow_internal_format = False
            print(f"[DEVICE] NPU 内部格式已禁用 (allow_internal_format=False)")
        except Exception:
            pass
        try:
            name = torch_npu.npu.get_device_name(0)
        except Exception:
            name = "Ascend NPU"
        print(f"[DEVICE] 华为昇腾NPU可用: {name}")
        return d, "npu"
    elif torch.cuda.is_available():
        d = torch.device("cuda:0")
        name = torch.cuda.get_device_name(0)
        print(f"[DEVICE] CUDA GPU可用: {name}")
        return d, "cuda"
    else:
        print("[DEVICE] 无加速器，使用CPU")
        return torch.device("cpu"), "cpu"


DEVICE, device_type = get_device()
DEVICE_STR = str(DEVICE)


def device_supports_pin_memory():
    """NPU 和 CUDA 都支持 pin_memory"""
    return device_type in ("cuda", "npu")


def empty_cache():
    """清理设备缓存"""
    if device_type == "npu" and _HAS_NPU:
        torch_npu.npu.empty_cache()
    elif device_type == "cuda":
        torch.cuda.empty_cache()


def synchronize():
    """同步设备流 (用于准确计时)"""
    if device_type == "npu" and _HAS_NPU:
        torch_npu.npu.synchronize()
    elif device_type == "cuda":
        torch.cuda.synchronize()


def manual_seed_all(seed: int):
    """在所有设备上设置随机种子"""
    torch.manual_seed(seed)
    if device_type == "npu" and _HAS_NPU:
        torch_npu.npu.manual_seed_all(seed)
    elif device_type == "cuda":
        torch.cuda.manual_seed_all(seed)


def get_scaler():
    """
    返回设备对应的 GradScaler (混合精度训练)。
    NPU: torch_npu.npu.amp.GradScaler()
    CUDA: torch.amp.GradScaler('cuda') — init_scale=128,
          GAN 训练初期损失较大, 默认 scale 会导致 fp16 梯度溢出 NaN
    CPU: None
    """
    if device_type == "npu" and _HAS_NPU:
        from torch_npu.npu import amp
        return amp.GradScaler()
    elif device_type == "cuda":
        return torch.amp.GradScaler('cuda', init_scale=128.0)
    else:
        return None


def autocast():
    """
    返回设备对应的 autocast 上下文管理器。
    注意: 本项目的 R1/R2 二阶梯度与 AMP fp16 不兼容, 因此实际均使用
    nullcontext (纯 fp32); 保留该接口以便后续按设备启用。
    """
    if device_type == "npu" and _HAS_NPU:
        return nullcontext()
    elif device_type == "cuda":
        return nullcontext()
    else:
        return nullcontext()


def use_amp():
    """是否启用 AMP (当前版本为纯 fp32 训练)"""
    return False


def scaler_unscale_(scaler, optimizer):
    """在梯度裁剪前执行 unscaling (GradScaler 场景)"""
    if scaler is not None:
        scaler.unscale_(optimizer)


def setup_matmul_precision():
    """设置矩阵乘法精度 (NPU/CUDA 启用 high 模式)"""
    if device_type in ("npu", "cuda"):
        torch.set_float32_matmul_precision('high')
        print(f"[DEVICE] float32 matmul precision 已设为 high")


def print_npu_env_hints():
    """打印 NPU 环境变量优化建议 (不自动设置, 避免副作用)"""
    if device_type != "npu":
        return
    import os
    hints = []
    if os.environ.get("TASK_QUEUE_ENABLE") is None:
        hints.append("  export TASK_QUEUE_ENABLE=2    # 任务队列: 减少算子下发延迟, 提升流水线效率")
    if os.environ.get("ASCEND_LAUNCH_BLOCKING") is None:
        hints.append("  export ASCEND_LAUNCH_BLOCKING=0  # 异步下发: kernel 不阻塞 Python 线程")
    if hints:
        print(f"[DEVICE] NPU 性能建议 (在 shell 中设置以下环境变量):")
        for h in hints:
            print(h)
        print(f"[DEVICE] 也推荐: export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True  # 减少内存碎片")


if __name__ == "__main__":
    print(f"设备: {DEVICE} (type={device_type})")
    print(f"支持 pin_memory: {device_supports_pin_memory()}")
    print(f"Has NPU: {_HAS_NPU}")
