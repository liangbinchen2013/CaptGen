"""
数据加载: 从 batch_* 目录读取验证码图片, 返回 image + label

- 图片: 35×90 灰度, 归一化到 tanh 空间 [-1, 1]
- 标签: 4 字符, 映射为大写 36 类 (0-9 + A-Z)
- NPU: 预加载到 CPU 内存 (约 1.26GB) + num_workers=0 (绕过 shm 限制)
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from config import *
from device_utils import device_supports_pin_memory, DEVICE


def label_to_tensor(label: str) -> torch.LongTensor:
    """将标签 'ab12' → [idx_a, idx_b, idx_1, idx_2], 自动应用大小写映射"""
    label = "".join(LABEL_CASE_MAP.get(c, c) for c in label)
    ids = [CHARS.index(c) for c in label]
    return torch.tensor(ids, dtype=torch.long)


def tensor_to_label(tensor: torch.Tensor) -> str:
    """将 [idx0, idx1, idx2, idx3] → 'AB12'"""
    ids = tensor.cpu().tolist()
    if isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return "".join(CHARS[i] if isinstance(i, int) else CHARS[i].argmax() for i in ids)


def parse_label_from_filename(filepath: Path) -> str:
    """从文件名 'captcha_XXXX[_xxxx].jpg' 中提取 4 字符标签"""
    stem = filepath.stem
    label_part = stem.replace("captcha_", "", 1)
    clean = "".join(c for c in label_part if c.isalnum())[:4]
    return clean


class CaptchaDataset(Dataset):
    """验证码数据集: 遍历 batch_* 目录下的所有图片"""

    def __init__(self, data_dir: Path, preload: bool = False, transform=None):
        self.data_dir = Path(data_dir)
        self.preload = preload
        self.transform = transform
        self.files = []
        self._cache = []      # 预加载缓存: [(tensor, label_tensor), ...]
        bad_count = 0

        for batch_dir in sorted(
            self.data_dir.glob("batch_*"),
            key=lambda p: int(p.name.split("_")[1])
        ):
            for f in batch_dir.glob("*.jpg"):
                try:
                    with Image.open(f) as im:
                        im.verify()
                    self.files.append(f)
                except Exception:
                    bad_count += 1

        if bad_count:
            print(f"[WARN] 跳过 {bad_count} 张损坏图片")

        # 预加载: 一次性读入内存 (NPU 大显存场景推荐)
        if self.preload:
            print(f"[PRELOAD] 正在预加载 {len(self.files)} 张图片到内存...")
            loaded = 0
            for f in self.files:
                try:
                    label = parse_label_from_filename(f)
                    label_tensor = label_to_tensor(label)
                    img = Image.open(f).convert("L")
                    img = np.array(img, dtype=np.float32)
                    img = (img / 127.5) - 1.0
                    img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
                    self._cache.append((img_tensor, label_tensor))
                    loaded += 1
                except Exception:
                    bad_count += 1
            self.files = self.files[:loaded]  # 只保留成功加载的
            if bad_count:
                print(f"[WARN] 预加载跳过 {bad_count} 张损坏图片")
            mem_mb = sum(t[0].numel() for t in self._cache) * 4 / 1e6
            print(f"[PRELOAD] 完成: {len(self._cache)} 张, 约 {mem_mb:.0f} MB")

        print(f"[OK] 加载 {len(self.files)} 张验证码图片")

    def __len__(self):
        return len(self._cache) if self.preload else len(self.files)

    def __getitem__(self, idx):
        # 预加载模式: 直接从内存返回
        if self.preload:
            return self._cache[idx]

        # 磁盘模式: 读取文件 (GPU/CPU)
        for attempt in range(10):
            img_path = self.files[idx]
            try:
                label = parse_label_from_filename(img_path)
                img = Image.open(img_path).convert("L")
                img = np.array(img, dtype=np.float32)
                img = (img / 127.5) - 1.0
                label_tensor = label_to_tensor(label)
                if self.transform:
                    img = self.transform(img)
                return torch.from_numpy(img).unsqueeze(0), label_tensor
            except Exception:
                idx = (idx + 1) % len(self.files)

        # 10 次重试后仍失败则抛出异常, 不静默返回零张量污染训练
        raise RuntimeError(
            f"Failed to load any valid image after 10 retries. "
            f"Last attempted: {self.files[idx] if idx < len(self.files) else 'N/A'}"
        )


def get_dataloader(data_dir=None, preload: bool = None,
                   batch_size: int = None) -> DataLoader:
    """创建 DataLoader

    Args:
        data_dir:   训练数据路径, 默认使用 config.DATA_DIR
        preload:    是否预加载到内存。None 则 NPU→True (绕过 shm), GPU/CPU→False
        batch_size: 批大小。None 则用 config.BATCH_SIZE
    """
    if data_dir is None:
        data_dir = DATA_DIR
    if batch_size is None:
        batch_size = BATCH_SIZE

    # NPU: 预加载到 CPU 内存 + num_workers=0, 绕过 /dev/shm 64MB 限制
    if preload is None:
        preload = (str(DEVICE).startswith("npu"))

    dataset = CaptchaDataset(data_dir, preload=preload)

    dl_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device_supports_pin_memory() and NUM_WORKERS == 0,
        drop_last=True,
    )
    # 仅在有 worker 时启用预取 (num_workers=0 时无效)
    if NUM_WORKERS > 0:
        dl_kwargs["prefetch_factor"] = 2
        dl_kwargs["persistent_workers"] = True

    return DataLoader(dataset, **dl_kwargs)


if __name__ == "__main__":
    loader = get_dataloader()
    imgs, labels = next(iter(loader))
    print(f"图像: {imgs.shape}, 标签: {labels.shape}")
    for i in range(4):
        print(f"  {tensor_to_label(labels[i])}")
