# B1 准入环境

## 可复现安装

```bash
conda create -n yolo_master_b1 python=3.11 -y
conda activate yolo_master_b1

git clone https://github.com/SidKC/YOLO-Master.git
cd YOLO-Master
git fetch origin tag rhino-2026-0824-b1-smoke
git switch --detach rhino-2026-0824-b1-smoke
# 官方增量验收基线为 Tencent/YOLO-Master
# acce839c7e895d6b179de7f7093fa879e237cc7b。
# Smoke 从同步后的 22140b74f1fdf7edc5ead18c3dbebce5e611c212 开始实现；
# 两者之间的 3 个上游提交不计入个人增量贡献。

pip install -r requirements.txt
pip install -e ".[dev]"
```

`requirements.txt`/`pyproject.toml` 已包含 `opencv-python`。旧环境若单独缺少 OpenCV，可在这个隔离环境中补装：

```bash
pip install "opencv-python>=4.6.0,!=4.13.0.90"
```

## 已执行环境

| 证据 | PyTorch/CUDA | 设备 | 备注 |
| --- | --- | --- | --- |
| 真实检测器 A100 集成 smoke | PyTorch 2.4.1+cu121 / CUDA 12.1 | NVIDIA A100-PCIE-40GB | batch 1 分别覆盖 expert 0/1，batch 2 同时覆盖两个专家；3/3 通过 |
| 4-seed 双专家组件 smoke | PyTorch 2.4.1+cu121 / CUDA 12.1 | NVIDIA A100-PCIE-40GB | 每 seed 160 steps；两个专家均反向并更新 |
| 真实检测器 CPU smoke | 已有隔离 PyTorch 环境，CPU-only | CPU | 逐个直接调用无 fixture 的 8 个测试函数；结果与 pytest 命令等价 |

## 复现

CPU 准入测试：

```bash
python scripts/b1/run_admission_smoke.py
```

一张可见 A100 上的真实 `DetectionModel` 集成测试：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/b1/run_a100_detector_smoke.py \
  --output runs/b1_a100_detector_smoke/result.json
```

本次 A100 实测可见显存为 39.49 GiB，预算上限取 80%（31.59 GiB），至少保留 7.90 GiB。RTX 4090 24 GiB 的兼容预算上限为 19.2 GiB、余量为 4.8 GiB，目前没有 4090 实测数据。

## 运行约束

- GPU 实验使用一进程一卡；不同硬件结果不合并。
- 不修改共享 Python 环境；缺失依赖安装到专用 conda 环境。
- 正式显存测量每次启动新进程，分开记录 cold/warm 路径。
- tiny smoke 的峰值只用于准入检查，不能替代完整模型训练的显存测量。
