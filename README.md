# ABH + DRC

This repository implements the crack segmentation method used in the paper. ABH produces foreground and background logits. DRC refines their difference near predicted boundaries using local and whole-image appearance references.

## Setup

The reference stack uses Linux x86-64, an NVIDIA GPU, Python 3.10, PyTorch 2.1.2 and CUDA 11.8. Install the CUDA 11.8 toolkit with a compatible C++ compiler, then run the following commands in the repository root.

```bash
conda env create -f environment.yml
conda activate abh-drc
export CUDA_HOME=/usr/local/cuda-11.8
python -m pip install --only-binary=mmcv-full 'mmcv-full==1.7.2' \
  --find-links https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html
python -m pip check
python third_party/build_cuda.py
```

For inference and frozen-cache export, `python third_party/build_cuda.py --inference-only` is sufficient. The builder downloads eight source files from a pinned SegMAN commit, verifies their SHA-256 hashes, overlays the three project-modified Oflex files, and builds Oflex. The full build also compiles the project resize operators used by ABH training. Downloaded sources and build outputs stay in ignored build paths.

See the [source lock](configs/source_provenance.json), [dependency notices](third_party/NOTICE.md), and [NATTEN wheel index](https://whl.natten.org/old/).

## Data and initialization

Choose `C500` (Crack500), `DC` (corrected DeepCrack), `CAM` (CamCrack789), `CM` (CrackMap), or `TUT`. Under the dataset root, place source images in `train/images`, `val/images`, and `test/images`, and the corresponding 8-bit masks in the matching `masks` directories. The ordered filename pairs in `splits/<dataset>.json` specify the exact membership. The training order files also record the repeated first sample for odd training split sizes. The loader applies bicubic resizing to 512 × 512, then binarizes masks at values greater than 127.

Supply the official SegMAN-L ImageNet EMA checkpoint or the corresponding encoder-only state dictionary. The default builds the zero-update decoder from seed 42. For the exact archived initialization, also supply the original `SEED42_ZERO_UPDATE_CORE.pth` with `--initial-core`. The archive applied this artifact to the freshly initialized common tensors before loading the ImageNet weights. It is an external input for exact reproduction.

## Training

```bash
python train.py abh --dataset DC --data-root data/DC \
  --encoder-weights weights/segman_l_ema.pth --output work/abh

python train.py cache --dataset DC --data-root data/DC \
  --abh-weights work/abh/best_abh.pth --output work/cache

python train.py drc --dataset DC --cache work/cache --output work/drc
```

Train ABH for 50 epochs and select its checkpoint by strict improvement in validation image-mean binary mIoU. Export frozen, unaugmented ABH scores for training and validation. Train only DRC for 20 epochs using soft-clDice, with its weight ramping up to 0.2 over five epochs. The zero-amplitude DRC head participates in validation selection at epoch zero. Both stages keep the earliest checkpoint on ties.

## Inference and paper evaluation

```bash
model=(--dataset DC --abh-weights work/abh/best_abh.pth --drc-weights work/drc/best_drc.pth)
python evaluate.py predict "${model[@]}" --image example.png --output work/example.npy
python evaluate.py validate "${model[@]}" --data-root data/DC --output work/validation.json
python evaluate.py test "${model[@]}" --data-root data/DC \
  --validation-lock work/validation.json --output work/test.json
```

Use the dataset that corresponds to the ABH checkpoint, including for single-image inference. The model applies ABH and DRC to each of the four rotations (0°, 90°, 180°, 270°), then inverse-rotates and averages the raw margins. Validation chooses one threshold per dataset from 0.00 to 0.99 in increments of 0.01, using per-image max normalization, uint8 conversion and image-mean binary mIoU. Test loads the threshold bound to the same dataset, split, weights and protocol. Structural metrics use a zero threshold on the fused raw margins.
