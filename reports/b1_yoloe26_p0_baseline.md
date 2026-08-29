# B1 YOLOE-26n P0 Open-Vocabulary Baseline

[中文版](b1_yoloe26_p0_baseline.zh-CN.md)

- **Status:** P0 complete
- **Completion date:** 2026-08-29
- **Source revision:** `4c2dada8a29ae235de44a4df757f2546658cf178`
- **Milestone tag:** `rhino-2026-0829-b1-yoloe26-p0-report-complete`

## Milestone overview

P0 is complete. The frozen YOLOE-26n path produced a loadable checkpoint, independent inference results, and COCO metrics for the 48 base classes, 17 new classes, and their 65-class union. The run therefore establishes a reproducible open-vocabulary detection baseline and closes the P0 execution loop.

In the frozen new-class evaluation, the model produced no new-class predictions and an AP of `0.00000`. Together with the base48 AP of `0.34494`, this establishes the baseline for subsequent work: the current training path learned the base classes, but did not yet produce effective detections for the new classes.

## Scientific question

On a fixed official YOLOE-26n implementation and checkpoint, can base-only COCO training produce a checkpoint that can be loaded in a separate process and evaluated on a frozen COCO 48/17 open-vocabulary split using real model outputs?

P0 used one model size, one seed, and one main training configuration. The focus was to establish a reproducible baseline and complete the full training, checkpoint, independent inference, and evaluation chain.

## Official YOLOE contract

The experiment follows the repository's [YOLOE documentation](../docs/en/models/yoloe.md): detection fine-tuning initializes the detection architecture from its YAML, loads the released same-scale segmentation checkpoint, and uses `YOLOEPETrainer`.

```python
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPETrainer

model = YOLOE("yoloe-26n.yaml")
model.load("yoloe-26n-seg.pt")
model.train(
    data="coco48_base.yaml",
    epochs=80,
    patience=10,
    imgsz=640,
    batch=16,
    optimizer="auto",
    amp=True,
    deterministic=True,
    seed=42,
    workers=4,
    cache=False,
    trainer=YOLOEPETrainer,
)
```

The run used full-parameter fine-tuning after prompt fusion; it did not use the separate linear-probing recipe. The global batch was 16 across four equal GPUs, or 4 samples per rank.

## Frozen evaluation protocol

The class partition follows the Bansal COCO 48/17 split used by [OVR-CNN](https://github.com/alirezazareian/ovr-cnn):

- Training retains annotations for the 48 base categories and removes images with no retained base annotation.
- The fused model vocabulary is ordered as 48 base classes followed by 17 new classes.
- The 17 new classes receive no positive training annotations but remain in the frozen 65-class output vocabulary.
- The remaining 15 COCO categories are excluded from training labels, output mapping, and evaluation.
- Evaluation runs on all 5,000 COCO `val2017` images with the original COCO category IDs.
- One prediction file is evaluated three times with `pycocotools.COCOeval(iouType="bbox")`: `overall65`, `base48`, and `new17`.

This report refers to the new-class metric as **new-class AP with a frozen 65-class vocabulary and zero positive labels**.

## Results

Training completed all 80 epochs in approximately `28.46 hours`, and both `best.pt` and `last.pt` were produced. A separate process loaded `best.pt`, verified the fused 65-class head, and completed a forward pass before full validation inference.

![P0 training curves](figures/b1_yoloe26_p0_training.svg)

*Figure 1: Training losses and validation metrics plotted from the [per-epoch training metrics CSV](b1_yoloe26_p0_results.csv) for all 80 epochs. The loss panel uses a logarithmic y-axis to show all three losses at their different scales.*

| Evaluation subset | Images | GT boxes | Predictions | AP | AP50 | AP75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall 65 classes | 5,000 | 33,152 | 551,220 | 0.25472 | 0.36249 | 0.27607 |
| Base 48 classes | 5,000 | 28,538 | 551,220 | 0.34494 | 0.49088 | 0.37384 |
| New 17 classes | 5,000 | 4,614 | 0 | 0.00000 | 0.00000 | 0.00000 |

![P0 COCO 48/17 evaluation results](figures/b1_yoloe26_p0_quality.svg)

*Figure 2: COCO box AP for overall65, base48, and new17, together with AP by object size.*

Independent inference processed the validation set at approximately `93.80 images/s`, with approximately `11.48 GiB` of peak allocated GPU memory.

![P0 execution stages and independent inference results](figures/b1_yoloe26_p0_execution.svg)

*Figure 3: Measured duration of training, checkpoint verification, and independent COCO evaluation, together with inference latency percentiles across 5,000 validation images.*

## P0 deliverables

- A 48/17 training configuration following the official YOLOE detection fine-tuning path.
- Results from all 80 training epochs, together with independently loadable `best.pt` and `last.pt` checkpoints.
- Overall65, base48, and new17 evaluation results covering the complete COCO `val2017` set.
- A fixed class order, COCO category-ID mapping, and evaluation procedure.
- A reusable training, checkpoint, independent inference, and COCO evaluation chain.

## Engineering fixes

Two engineering fixes were completed during execution:

- The trainer was selected directly as `YOLOEPETrainer`; a duplicate trainer override was removed.
- Long validation inference was changed to bounded chunks with predictor reset between chunks, and the COCO evaluator was isolated from a conflicting package installation.

## Result interpretation

The clearest P0 observation is the difference between the base and new classes: base48 AP reached `0.34494`, while new17 produced no predictions. This establishes a clear baseline state—the official YOLOE-26n detection fine-tuning path learned the base classes under the current 48/17 setting, while new-class detection remained empty. Subsequent work can investigate how training objectives, vocabulary fusion, and conditional compute relate to this difference.

## Reproduction information

The reproduction anchors for this result are the source revision at the top of this report, the official YOLOE training contract, the training configuration above, the frozen COCO 48/17 policy, and standard COCO bounding-box evaluation. The exact class-name and category-ID mapping was materialized as a machine-readable split file before training and kept unchanged throughout training and evaluation.

The report figures are generated by `scripts/plot_b1_yoloe26_p0.py` directly from the run receipts and training metrics:

```bash
python scripts/plot_b1_yoloe26_p0.py \
    --main-receipt <MAIN_RECEIPT.json> \
    --verify-receipt <VERIFY_RECEIPT.json> \
    --eval-receipt <INFERENCE_EVAL_RECEIPT.json> \
    --training-csv reports/b1_yoloe26_p0_results.csv \
    --output-dir reports/figures
```
