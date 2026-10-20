# HiVa-DETR

Official code and checkpoints of **Fine Values, Coarse Tokens: High-Resolution Value Sampling for Detecting Tiny Objects with DETRs**.

High-Resolution Value Sampling (HiVa) gives the decoder of a DETR a stride-2 feature map (P1) that is read as values only.
P1 is built from the backbone's stem and the stride-4 pyramid level, and every decoder layer samples it in its deformable cross-attention.
It never enters the encoder or the query selection, so the number of tokens is unchanged.
HiVa-DETR applies HiVa to D-FINE.

This repository currently contains the inference and evaluation code and the trained checkpoints.

## Results and checkpoints

**VisDrone2019-DET** (validation, 800 x 800)

| Model | Params (M) | GFLOPs | AP | AP50 | AP75 | APs | APm | APl | Checkpoint |
|---|---|---|---|---|---|---|---|---|---|
| HiVa-DETR-S | 12.0 | 130.3 | 37.6 | 59.9 | 39.5 | 30.2 | 47.4 | 61.4 | [hiva_detr_s_visdrone.pth](https://github.com/shijie-research/HiVa-DETR/releases/download/v1.0/hiva_detr_s_visdrone.pth) |
| HiVa-DETR-M | 22.8 | 263.9 | 39.7 | 62.8 | 41.4 | 32.4 | 49.5 | 63.4 | [hiva_detr_m_visdrone.pth](https://github.com/shijie-research/HiVa-DETR/releases/download/v1.0/hiva_detr_m_visdrone.pth) |
| HiVa-DETR-L | 35.2 | 349.5 | 39.9 | 63.1 | 42.1 | 32.5 | 49.9 | 61.5 | [hiva_detr_l_visdrone.pth](https://github.com/shijie-research/HiVa-DETR/releases/download/v1.0/hiva_detr_l_visdrone.pth) |

**AI-TOD-v2** (test)

| Model | Params (M) | GFLOPs | AP | AP50 | AP75 | APvt | APt | APs | APm | Checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|
| HiVa-DETR-S | 12.0 | 128.0* | 32.8 | 67.8 | 27.9 | 17.1 | 32.5 | 39.5 | 46.5 | [hiva_detr_s_aitod.pth](https://github.com/shijie-research/HiVa-DETR/releases/download/v1.0/hiva_detr_s_aitod.pth) |
| HiVa-DETR-M | 22.8 | 260.6* | 33.9 | 67.7 | 29.8 | 17.1 | 34.0 | 39.5 | 47.9 | [hiva_detr_m_aitod.pth](https://github.com/shijie-research/HiVa-DETR/releases/download/v1.0/hiva_detr_m_aitod.pth) |
| HiVa-DETR-L | 35.2 | 344.8* | 34.7 | 69.4 | 30.4 | 18.7 | 34.7 | 39.6 | 49.3 | [hiva_detr_l_aitod.pth](https://github.com/shijie-research/HiVa-DETR/releases/download/v1.0/hiva_detr_l_aitod.pth) |

\* Averaged over the test split, as the number of queries varies per image.

**TinyPerson** (test, AP50)

| Model | tiny1 | tiny2 | tiny3 | tiny | small | Checkpoint |
|---|---|---|---|---|---|---|
| HiVa-DETR-S | 46.15 | 61.02 | 65.77 | 57.68 | 68.26 | [hiva_detr_s_tinyperson.pth](https://github.com/shijie-research/HiVa-DETR/releases/download/v1.0/hiva_detr_s_tinyperson.pth) |

Results may differ in the last digit on other GPUs.

## Installation

Python 3.10 or later.

```bash
pip install -r requirements.txt
```

## Data

The datasets are read from the Hugging Face Hub and downloaded on first use.

| Dataset | Hub repository |
|---|---|
| VisDrone2019-DET | [shijli/visdrone2019-det](https://huggingface.co/datasets/shijli/visdrone2019-det) |
| AI-TOD-v2 | [shijli/aitod-v2](https://huggingface.co/datasets/shijli/aitod-v2) |
| TinyPerson | [shijli/tinyperson](https://huggingface.co/datasets/shijli/tinyperson) |

AI-TOD-v2 and TinyPerson are gated.
Accept their terms on the Hub and log in with `huggingface-cli login` before the first run.

## Evaluation

Download a checkpoint into `weights/` and run

```bash
python test.py -c configs/HiVa-DETR-S-VisDrone.yml -r weights/hiva_detr_s_visdrone.pth
python test.py -c configs/HiVa-DETR-S-AITOD.yml -r weights/hiva_detr_s_aitod.pth
python test.py -c configs/HiVa-DETR-S-TinyPerson.yml -r weights/hiva_detr_s_tinyperson.pth
```

Config entries can be overridden on the command line, e.g. `-u val_dataloader.total_batch_size=8 val_dataloader.num_workers=2`.

## AI assistance

The research idea, the method and the design of the experiments are the authors' own.
The code was written, refactored and organized with [Claude Code](https://claude.com/claude-code) (Anthropic's Claude Opus models) under the authors' direction.
The authors reviewed the code and checked the released checkpoints against the reported results.

## Acknowledgments

This code is built on [D-FINE](https://github.com/Peterande/D-FINE), [RT-DETR](https://github.com/lyuwenyu/RT-DETR) and [Dome-DETR](https://github.com/RicePasteM/Dome-DETR).
The TinyPerson evaluation follows [TinyBenchmark](https://github.com/ucas-vg/TinyBenchmark).

## License

Apache License 2.0. See [LICENSE](LICENSE).
