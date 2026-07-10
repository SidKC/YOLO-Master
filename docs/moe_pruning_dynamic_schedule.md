# MoE 专家剪枝与动态调度

本文说明 Issue #52 的可复现实验流程，以及为 MoE 训练新增的动态超参数调度策略。

## 动态平衡系数调度

该调度器默认关闭。启用后，训练器会在一个 epoch 内累计所有 batch 的专家利用率，计算各层 Gini 系数的均值，并按下式更新平衡损失系数：

```text
ema_t = beta * ema_(t-1) + (1 - beta) * gini_t
coeff_(t+1) = clip(base_coeff * exp(alpha * (ema_t - target_gini)),
                   min_balance_coeff,
                   max_balance_coeff)
```

当路由集中到少数专家时，Gini 系数上升，平衡损失随之增强；当路由已经较均衡时，系数会降低，为专家分化保留空间。epoch 累计器仅存在于当前进程，不写入 checkpoint。调度功能保持向后兼容，可按以下方式运行三组对照：

```bash
python scripts/run_moe_dynamic_schedule_ablation.py \
  --variant all --model ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml \
  --data VisDrone.yaml --epochs 10 --device 0 --no-amp
```

三组分别为固定平衡系数、Gini 动态调度和固定低平衡系数消融。脚本会在构建每个模型前重置随机种子，为每个 variant 单独记录初始模型状态，并在 summary 阶段校验 SHA-256 与运行配置。动态组每个 epoch 都会在 JSONL 和 `results.csv` 中记录 opportunity、event、nontrivial action、参与层数和路由观测次数。

调度状态在 epoch 通过验证与 recovery 检查后提交。发生 recovery 时，本轮路由累计会被清空，调度器和 trace 保持在上一个 accepted epoch。汇总阶段生成 `dynamic_schedule_trace_audit.json`，核对 `results.csv` 与 trace 的 epoch、观测数和事件计数。当前累计器按单进程工作，动态模式暂不支持 DDP；默认训练路径不受影响。

## 专家剪枝阈值实验

先在训练集上生成不可变的校准信号。显式物化确定性子集，可以避免验证路径忽略 `fraction` 参数：

```bash
python scripts/diagnose_moe_pruning_signal.py \
  --model runs/train/esmoe_n/weights/best.pt \
  --data VisDrone.yaml --split train --fraction 0.10 --seed 42 \
  --output runs/moe_signal/train_calibration.json
```

然后生成 Issue 要求的五个逻辑阈值实验计划：

```bash
python scripts/moe_pruning_sweep.py \
  --model runs/train/esmoe_n/weights/best.pt \
  --signal-json runs/moe_signal/train_calibration.json \
  --calibration-dataset VisDrone.yaml \
  --eval-dataset VisDrone.yaml \
  --train-dataset VisDrone.yaml \
  --thresholds 0.05 0.10 0.15 0.20 0.30 \
  --out-dir runs/moe_pruning_sweep
```

脚本会生成：

- `moe_pruning_sweep_manifest.json`：记录每个阈值的准确命令与 provenance；
- `moe_pruning_sweep.csv`：保留直接推理与 LoRA 10-epoch 恢复两类逻辑实验行。

如果多个逻辑阈值得到相同的逐层专家 signature，脚本只执行一次物理剪枝和一次恢复训练，但仍在结果表中保留所有逻辑阈值。剪枝器还会校验 signal 中的模型 SHA-256 是否与输入 checkpoint 一致，不一致时拒绝执行。

完成质量评估和双顺序延迟测量后，使用 measurement spec 组装完整的 5×2 结果表。组装器会检查样本数、路由事件、checkpoint hash 和保留专家结构，只有全部通过才接受该结果行。

```json
{
  "measurements": {
    "0.05": {
      "direct": {
        "eval_json": "baseline_eval.json",
        "latency_jsons": ["order_ab.json", "order_ba.json"]
      },
      "lora10": {
        "eval_json": "baseline_lora10_eval.json",
        "latency_jsons": ["lora_order_ab.json", "lora_order_ba.json"],
        "train_results_csv": "baseline_lora10/results.csv"
      }
    },
    "0.30": {
      "direct": {
        "eval_json": "pruned_eval.json",
        "latency_jsons": ["order_ab.json", "order_ba.json"]
      },
      "lora10": {
        "eval_json": "pruned_lora10_eval.json",
        "latency_jsons": ["lora_order_ab.json", "lora_order_ba.json"],
        "train_results_csv": "pruned_lora10/results.csv"
      }
    }
  }
}
```

```bash
python scripts/assemble_moe_pruning_results.py \
  --manifest runs/moe_pruning_sweep/moe_pruning_sweep_manifest.json \
  --measurements runs/moe_pruning_sweep/measurements.json \
  --output runs/moe_pruning_sweep/moe_pruning_results.csv
```

## 绘图与推荐门槛

```bash
python scripts/plot_moe_pruning_sweep.py \
  runs/moe_pruning_sweep/moe_pruning_results.csv
```

绘图脚本会生成阈值曲线、质量—延迟—资源三目标 Pareto 前沿及机器可读分析文件。共享同一物理结构的重复阈值会在图中合并标注，但仍保留在原始 5×2 CSV 中。no-op baseline 作为参考散点单独输出，只有数学上非支配的点进入 `pareto_front.csv` 和前沿连线。推荐点需要同时满足 mAP 降幅、保留专家结构和测量完整性约束；没有点通过门槛时，结果标记为“未观察到 Sweet Spot”。

## 动态调度对照与副作用

建议至少比较以下三组：

- 固定基线：关闭动态调度；
- 动态组：根据整 epoch Gini 调整平衡系数；
- 消融组：使用固定的低平衡系数。

Issue 定义的收敛加速比为：

```text
speedup = baseline_epoch_to_95pct_final_map / experiment_epoch_to_95pct_final_map
```

分析时还要检查 late-stage collapse、Gini 振荡、过度平衡和最终 mAP 下降。若训练保持稳定，可继续比较较小的 `alpha`、更大的 `beta` 或不同的 `target_gini`。

## VisDrone 10-Epoch 独立复现结果

三组单卡实验使用相同 seed、数据集、batch size、输入尺寸和经过校验的初始模型状态 hash。修正后的动态组恰好生成 10 条 accepted-epoch trace，对应 epoch 1 至 10。每条记录包含 3,236 次路由观测，即四个 MoE 层各 809 次；最终 `opportunity/event/nontrivial-action` 计数为 `10/10/10`。

| 实验组 | 平衡策略 | 最佳 mAP50-95 | 最佳 epoch | 最终 mAP50-95 |
| --- | --- | ---: | ---: | ---: |
| 固定基线 | `1.0` | 0.03551 | 4 | 0.00750 |
| Gini 动态调度 | EMA 指数调度 | 0.04144 | 6 | 0.00905 |
| 固定低系数消融 | `0.3` | 0.03950 | 4 | 0.00493 |

动态组的平均/最终 Gini 为 `0.04615/0.01058`，平均/最终平衡系数为 `0.82399/0.80416`。三组均出现严重的 late-stage quality collapse：最终 mAP50-95 仅为各自最佳值的 21.1%、21.8% 和 12.5%。由于固定基线的最终值已经坍塌，三组都在 epoch 1 达到“固定基线最终精度的 95%”，形式上的收敛比例均为 `1.0`，但该指标不再具有科学区分力。

动态组在该单 seed 短实验中的最佳值和最终值均高于固定基线，但现有证据不能证明收敛加速，也不能证明稳定的最终质量收益。在推荐该调度策略之前，仍需稳定的长周期基线、多随机种子，以及不依赖 collapsed-final checkpoint 的预注册收敛目标。
