# EEEM066 — Knife Classification (Fine-Grained Image Classification)

Coursework for **EEEM066 Fundamentals of Machine Learning** (University of Surrey).
The project fine-tunes ImageNet-pretrained CNNs from [`timm`](https://github.com/huggingface/pytorch-image-models)
on the **EEEM066_KnifeHunter** dataset — a 543-class fine-grained knife classification
problem — and studies how the optimizer, learning-rate schedule and data augmentation
affect validation and test mAP.

The default configuration is **EfficientNet-B0**, trained for 10 epochs with Adam and a
cosine-annealed learning rate.

---

## Reproducing this on your machine

### 1. Prerequisites

| Requirement | Detail |
| :--- | :--- |
| Python | 3.8 – 3.10 (the pinned `torch==1.12.1` has no wheels for 3.11+) |
| GPU | **An NVIDIA GPU with CUDA is required.** See the note below. |
| Disk | ~2 GB for the dataset, plus ~20 MB per saved checkpoint |

> **The training and testing code is CUDA-only.** `Training.py` and `Testing.py` call
> `.cuda()` on every batch, build the loss with `nn.CrossEntropyLoss().cuda()`, and use
> `torch.cuda.amp` for mixed precision. These calls are unconditional, so the scripts will
> raise an error on a CPU-only or Apple-Silicon (MPS) machine even though the code also
> computes a `device` variable. Running without an NVIDIA GPU requires editing the scripts.

### 2. Get the code

```bash
git clone https://github.com/Sayzal28/FML.git
cd FML
```

### 3. Create the environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins the versions the results were produced with:

```
numpy==1.21.5   opencv_python_headless==4.7.0.72   pandas==1.4.4   Pillow==10.0.1
scikit_learn==1.0.2   timm==0.6.12   torch==1.12.1   torchvision==0.13.1
```

If `pip` cannot find a `torch==1.12.1` wheel for your CUDA version, install it from the
PyTorch index first, then install the rest:

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

### 4. Get the dataset

The **EEEM066_KnifeHunter** dataset is distributed by the module organisers and is *not*
included in this repository. Obtain it from the EEEM066 coursework materials.

By default the scripts expect it as a **sibling of this repository**:

```
parent-directory/
├── FML/                      # this repository
└── EEEM066_KnifeHunter/      # the dataset
    ├── Train/<class_name>/*.jpg
    ├── Validation/<class_name>/*.jpg
    └── Test/<class_name>/*.jpg
```

If you put it elsewhere, change `--dataset_location` in `train.sh` and `test.sh`.

The splits are already defined in this repo and are read as CSVs of `Id,Label` pairs,
where `Id` is a path relative to `--dataset_location`:

| File | Rows | Example `Id` |
| :--- | ---: | :--- |
| `dataset/train.csv` | 25,250 | `Train/BForce1/BForce1_1.jpg` |
| `dataset/validation.csv` | 440 | `Validation/BForce1/BForce1_1.jpg` |
| `dataset/test.csv` | 469 | `Test/BForce1/BForce1_1.jpg` |
| `dataset/classes.csv` | 543 | label → class-name lookup |

### 5. Train

```bash
bash train.sh
```

This runs the baseline configuration:

```
--model_mode tf_efficientnet_b0   --n_classes 543     --epochs 10
--batch_size 32                   --learning_rate 0.00005
--optim adam                      --lr-scheduler CosineAnnealingLR
--resized_img_weight 224          --resized_img_height 224
--brightness 0.2 --contrast 0.2 --saturation 0.2 --hue 0.2
--seed 0                          --saved_checkpoint_path Knife-Effb0
```

**Outputs**

* `Knife-Effb0/Knife-tf_efficientnet_b0-E{1..10}.pth` — one checkpoint per epoch.
* `logs/log_train_<timestamp>.txt` — the full argument dump plus a per-epoch table of
  training loss and validation mAP.

Progress is printed live as a single updating line per iteration:

```
| Mode  |  Iter  | Epoch  |   Loss   |    mAP   |     Time     |
| train | 394.0 |   9.0 |    0.052 |   0.505 |  8 min 08 sec |
```

> `--saved_checkpoint_path` is created with `os.mkdir`, not `os.makedirs`. Use a
> single-level directory name (`Knife-Effb0`), or create the parent directories yourself
> first — a nested path such as `runs/exp1/ckpts` will fail.

### 6. Test

`test.sh` evaluates the **epoch-10** checkpoint by default:

```bash
bash test.sh
```

If you trained for a different number of epochs, update `--model-path` to match the
checkpoint you want (`Knife-Effb0/Knife-tf_efficientnet_b0-E<N>.pth`).

The score is written to `logs/log_test_<timestamp>.txt`:

```
 | Mode  |    mAP   |     Time     |
| test  |   0.478 |  0 min 00 sec |
```

### 7. Expected result

Training the baseline as shipped reproduces roughly:

| Metric | Value |
| :--- | :--- |
| Validation mAP (epoch 10) | ≈ 0.50 |
| **Test mAP** | **≈ 0.478** |
| Wall-clock training time | ≈ 8 minutes (10 epochs, single GPU) |

Exact numbers vary slightly with GPU model and cuDNN version even with `--seed 0`, because
the data loader uses 8 worker processes and cuDNN autotuning is non-deterministic.

The `logs/` directory in this repository contains the original runs from the report, so
you can compare your output against them directly.

---

## Running other configurations

All hyperparameters are command-line arguments defined in `args.py`. The simplest way to
run a variant is to copy `train.sh` and edit the flags.

**Backbone** — any `timm` model name works, as long as `--n_classes` stays 543:

```bash
--model_mode tf_efficientnet_b1      # or resnet50, convnext_tiny, ...
```

**Optimizer** (`src/optimizers.py`) — one of:

`adam` · `amsgrad` · `sgd` · `rmsprop`

Related flags: `--learning_rate`, `--weight-decay`, `--momentum`, `--sgd-dampening`,
`--sgd-nesterov`, `--rmsprop-alpha`, `--adam-beta1`, `--adam-beta2`.

**LR scheduler** (`src/lr_schedulers.py`) — one of:

`single_step` · `multi_step` · `CosineAnnealingLR`

Related flags: `--stepsize` (accepts multiple values), `--gamma`.

**Augmentation** — applied only to the training split, in this order:

```bash
--brightness 0.2 --contrast 0.2 --saturation 0.2 --hue 0.2   # ColorJitter
--random_rotation 15                                          # degrees, 0 disables
--horizontal_flip 0.5 --vertical_flip 0.5                     # probabilities, 0 disables
--color-aug                                                   # RGB intensity jitter
--random-erase                                                # random erasing
```

Validation and test always use plain resize → tensor → ImageNet normalisation.

---

## Repository layout

```
Training.py           Training entry point (train + validate loop, saves one ckpt/epoch)
Testing.py            Evaluation entry point (loads a checkpoint, reports test mAP)
args.py               Every command-line argument, plus optimizer/scheduler kwarg builders
data.py               knifeDataset — reads the CSV splits and builds the transforms
utils.py              AverageMeter, Logger, mAP/top-k accuracy, seeding, log formatting
src/optimizers.py     init_optimizer  — adam / amsgrad / sgd / rmsprop
src/lr_schedulers.py  init_lr_scheduler — single_step / multi_step / CosineAnnealingLR
src/transforms.py     ColorAugmentation and RandomErasing
train.sh / test.sh    The baseline configurations
dataset/*.csv         Train / validation / test splits and the class lookup
logs/                 Original training and test logs from the report
```

### A note on `args.py` and imports

`data.py` calls `argument_parser().parse_args()` **at import time**. Any script that
imports `data.py` therefore parses `sys.argv` immediately and will fail if the required
`--dataset_location` argument is missing. Always launch through `Training.py` or
`Testing.py` with the full flag set rather than importing `data` from a notebook or REPL.

### `STUDENT_ID` / `STUDENT_NAME`

`train.sh` and `test.sh` set these environment variables. They are only recorded in the
log header for coursework submission and have no effect on training. Leaving the
placeholder values in place is fine.

---

## Attribution

Forked from [`Surrey-EEEM066/CSWK_2025-26`](https://github.com/Surrey-EEEM066/CSWK_2025-26),
the EEEM066 coursework template. The training and evaluation harness comes from that
template; the experiments, configurations and analysis are this project's own.

Author: Saizalpreet Kaur (6915210) — see `EEEM066-6915210-Assignment.pdf` for the report.
