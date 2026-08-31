# B1 P1-2 文本条件路由实验报告

[English](b1_p1_2_text_causal.md)

- **状态：** P1-2 已完成
- **报告日期：** 2026-08-31
- **实现版本：** `47ac2ecd27a4278aee7bda34a63ab108d4719e65`
- **实验协议：** 单 seed、每组 20 个训练步骤、完整 COCO `val2017` 评测

## 结果概述

P1-2 在已发布的 unfused YOLOE-26n detector 上完成了 `true-text` 与 `zero-text` 配对实验。两组使用完全相同的 detector core、开放词汇 classifier、adapter 初始化、图像顺序、optimizer、loss 和训练预算，唯一的实验变量是 Router condition。

真实 Router 文本改变了 Router logits，使 40 次配对机会中的 10 次发生 hard Top-1 专家切换，并使固定 probe detector output 的 650,000 个元素中有 66,725 个发生变化。两组训练后的 adapter 随后完成了相同的 5,000 图像 COCO 评测。

在 17 个 new 类上，`true-text` AP 为 `0.19183`，`zero-text` AP 为 `0.16504`。配对差值为 `+0.02679`，即 2.68 个 AP points，相对于 zero-text 结果提升约 16.2%。在 overall65 和 base48 上，AP、AP50 与 AP75 也都得到正向差值。

## 实验问题

在固定已发布 unfused 开放词汇 detector、adapter 初始化、数据顺序和 20 步训练预算的条件下，真实 Router 文本相对于全零条件，是否会改变路由并将影响传导到 detector 输出？

在观察配对结果之前，实验为这个问题固定了两个读数：非零的配对 Router-logit 效应，以及 classifier 保持不变时非零的固定 probe detector-output 效应。

## 已完成实现

P1-2 在 detector 的 P5 特征层接入了一个可选的文本条件 adapter：

- `TextConditionedMoT` 包含两个专家，采用 sample-level hard Top-1 路由；
- Router condition 与 classifier 文本嵌入使用相互独立的输入，使干预前后的 65 类 detector 词表保持固定；
- 已发布的 YOLOE detector core 和 classifier 全程冻结，并保持字节级一致；
- optimizer 只接收 adapter 参数；
- routing auxiliary loss 在每个训练步骤发布并消费一次，同时记录重复和过期消费；
- adapter-off 加载保持已发布模型结构和 checkpoint key 布局一致。

## 配对实验协议

两组实验的唯一区别是 Router condition：

- **true-text：** 由固定 schedule 选出的冻结真实 prompt condition；
- **zero-text：** 与真实条件具有相同 shape、dtype、device 和 batch expansion 的全零 tensor。

两组共同使用相同的已发布 unfused YOLOE-26n parent、字节级一致的初始 adapter state、seed `0`、20 个 optimizer steps、batch size `2`、image size `320`，以及 learning rate 为 `0.001`、weight decay 为零的 AdamW。全部 20 步均采用 hard Top-1 routing。Image ID、condition schedule、optimizer、loss 和样本顺序严格配对。两组在单张 RTX 4090 上串行执行，不使用 DDP。

正式评测使用相同的冻结 65 类 classifier 和全部 5,000 张 COCO `val2017` 图像。两组统一采用 image size `640`、confidence threshold `0.001`、IoU threshold `0.7`、maximum detections `300` 和 batch size `1`。

### 65 类评测划分

本报告沿用冻结的 Bansal/OVR-CNN COCO 48/17 类别划分。图中的三组指标不是三个模型或三次独立实验，而是对同一 65 类 classifier、同一份预测结果进行的三个 COCOeval 视角：

- **Base 48：** 48 个 base/seen 类。在该冻结协议中，这些类别提供正训练标注；该指标用于观察模型在训练可见类别上的检测质量。
- **New 17：** 17 个 new/novel/unseen 类。它们不提供正训练标注，但保留在同一个冻结的 65 类输出词表中；该指标用于衡量模型对训练未见类别的开放词汇泛化。
- **Overall 65：** Base 48 与 New 17 的并集，即这套开放词汇任务实际评测的全部 65 类；该指标表示模型在整个冻结任务上的总体检测质量。

因此，`Overall 65 = Base 48 + New 17` 指的是类别集合关系。Overall 65 AP 是在 65 类并集上重新执行 COCOeval 得到的，并不是 Base 48 AP 与 New 17 AP 的简单算术平均。其余 15 个 COCO 类不进入本协议的训练标签、输出映射或评测集合，所以 Overall 65 也不表示完整的 COCO 80 类。

## 从路由到检测结果的证据

两组都完成了 20/20 个 optimizer steps。两个专家均获得非零 task gradient 和参数更新。每组 auxiliary publish/consume 均为 `20/20`，重复和 stale record 均为零。Detector core 与 classifier 保持字节级一致，初始 adapter state 和配对样本顺序也完全相同。

配对 Router-logit 距离为 L1 `2.43882`、L2 `0.35561`。Hard Top-1 selection 在 `10/40` 次机会中发生变化，切换率为 `25%`。固定 probe output 的变化元素数为 `66,725/650,000`，占 `10.27%`；L2 距离为 `4,196.37`，Linf 距离为 `313.51`。

## 完整 COCO 评测

两个训练后的 adapter 使用相同的冻结 65 类 classifier，在全部 5,000 张 COCO `val2017` 图像上完成评测。

![True-text 与 zero-text 的 COCO 指标](figures/b1_p1_2_text_causal_quality.svg)

*图 1：同一份 65 类预测在全部任务类别（Overall 65）、训练可见类别（Base 48）和训练未见类别（New 17）上的 COCO box AP、AP50 与 AP75。图中全部配对差值均为 true-text 更高。*

AP 差值在 overall65 上为 `+0.01875`，在 base48 上为 `+0.01590`，在 new17 上为 `+0.02679`。New17 的 AP50 差值为 `+0.03765`，AP75 差值为 `+0.03146`。

## New-class 参考坐标

已发布的 adapter-off substrate 也使用相同的冻结 65 类词表完成了评测，new17 AP 为 `0.32761`。图 2 将 P1-2 已完成的配对测量放在这个既有 new-class 坐标中展示。

![New17 AP 参考坐标](figures/b1_p1_2_text_causal_new17_context.svg)

*图 2：两个 P1-2 Router condition 与已发布 adapter-off reference 的 new17 AP；括号标出了 true-text 相对于 zero-text 的实测差值。*

## 已完成产物

- 接入真实 YOLOE detector 的两专家文本条件 P5 adapter；
- 覆盖 Router condition、routing auxiliary-loss 消费、冻结组件、checkpoint 兼容性和 detector 集成的 contract tests；
- 初始状态与样本顺序严格一致的 20 步 `true-text`、`zero-text` 配对训练结果；
- 两个训练后 adapter 在完整 5,000 图像上的 overall65、base48、new17 COCO 评测；
- 一份机器可读的[结果摘要](b1_p1_2_text_causal_results.csv)和可复现制图脚本。

## 工程记录

第一次正式评测在推理开始前因隔离 evaluator 缺少依赖而停止。随后通过 run-local evaluator overlay 提供已经验证的依赖，第二次尝试完成了冻结的评测协议。两次尝试及其终态 receipts 均已分别保存。

## 复现信息

本次结果的公开复现锚点包括文首的实现版本、配对训练设置、固定的 COCO 评测协议和机器可读结果摘要。原始 predictions、checkpoints 和执行 receipts 已保存在实验存储中。

报告中的图片由结果摘要直接生成：

```bash
python scripts/plot_b1_p1_2_text_causal.py \
    --results-csv reports/b1_p1_2_text_causal_results.csv \
    --output-dir reports/figures
```
