"""Shared configuration, image preprocessing and paper metrics."""
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATASETS = ("C500", "DC", "CAM", "CM", "TUT")
IMAGE_SIZE = (512, 512)
RGB_MEAN = [123.675, 116.28, 103.53]
RGB_STD = [58.395, 57.12, 57.375]
QUARTER_TURNS = (0, 1, 2, 3)
PROTOCOL = "abh-frozen-drc-rot4-val-threshold-v1"
_augment = None

MODEL_CONFIG = {'backbone': {'pretrained': None, 'style': 'pytorch', 'type': 'SegMANEncoder_l'},
 'decode_head': {
     'type': 'PlainSecondHeadDecoder',
     'align_corners': False,
     'boundary_width': 3,
     'channels': 224,
     'dropout_ratio': 0.1,
     'feat_proj_dim': 432,
     'in_channels': [96, 192, 432, 640],
     'in_index': [0, 1, 2, 3],
     'loss_decode': {'loss_weight': 1.0, 'type': 'CrossEntropyLoss', 'use_sigmoid': False},
     'norm_cfg': {'requires_grad': True, 'type': 'SyncBN'},
     'num_classes': 2,
 },
 'pretrained': None,
 'test_cfg': {'mode': 'whole'},
 'train_cfg': {},
 'type': 'EncoderDecoder'}

OPTIMIZER_CONFIG = {'betas': (0.9, 0.999),
 'lr': 6e-05,
 'paramwise_cfg': {'custom_keys': {'head': {'lr_mult': 10.0},
                                   'norm': {'decay_mult': 0.0},
                                   'pos_block': {'decay_mult': 0.0}}},
 'type': 'AdamW',
 'weight_decay': 0.01}

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False))
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(value,temporary)
    temporary.replace(path)


def statehash(model):
    h = hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        h.update(key.encode()+value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def seed():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rng():
    import torch
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),
                numpy=np.random.get_state(),python=random.getstate())


def restore(value):
    import torch
    torch.set_rng_state(value['torch'])
    torch.cuda.set_rng_state_all(value['cuda'])
    np.random.set_state(value['numpy'])
    random.setstate(value['python'])


def normalize(bgr):
    import mmcv
    rgb = mmcv.imnormalize(
        bgr.astype(np.float32), np.array(RGB_MEAN, np.float32),
        np.array(RGB_STD, np.float32), to_rgb=True,
    )
    return rgb.transpose(2, 0, 1).copy()


def export(z):
    z = np.asarray(z, dtype=np.float32)
    maximum = float(z.max())
    return np.zeros_like(z, np.uint8) if maximum <= 1e-12 else np.clip(255. * (z / maximum), 0, 255).astype(np.uint8)


def curve(score, label):
    """Exact uint8 threshold counts; same degeneracy handling as frozen RIFT."""
    gt = (label / 255).astype("uint8")
    pos = np.bincount(score[gt == 1], minlength=256).astype(np.int64)
    neg = np.bincount(score[gt == 0], minlength=256).astype(np.int64)
    decisions = (np.arange(256, dtype=np.float64)[None, :] / 255 >
                 np.arange(0., 1., .01)[:, None]).astype(np.int64)
    tp, fp = decisions @ pos, decisions @ neg
    fn, tn = pos.sum() - tp, neg.sum() - fp
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (tp / (fn + fp + tp) + tn / (fn + fp + tn)) / 2
    result[fn + fp + tp <= 0] = 0.
    return result


def aggregate(curves):
    a = np.asarray(curves, dtype=np.float64)
    best, threshold = -np.inf, None
    for i, t in enumerate(np.arange(0., 1., .01)):
        value = np.mean(np.ascontiguousarray(a[:, i]))
        if value > best:
            best, threshold = float(value), float(t)
    return best, threshold


def select_on_validation(margins_and_labels):
    curves = [curve(export(margin), label * 255) for margin, label in margins_and_labels]
    score, threshold = aggregate(curves)
    if not np.isfinite(score):
        raise RuntimeError("Validation did not yield a finite mIoU")
    return {"mIoU": score, "threshold": threshold, "samples": len(curves)}


def image_miou(prediction, target):
    """One threshold, with the locked evaluator's empty foreground convention."""
    tp = np.count_nonzero(prediction & target)
    tn = np.count_nonzero(~prediction & ~target)
    mismatch = target.size - tp - tn
    foreground_union = tp + mismatch
    if foreground_union == 0:
        return 0.0
    background_union = tn + mismatch
    if background_union == 0:
        raise ValueError("Binary mIoU is undefined for an empty background union")
    return 0.5 * (tp / foreground_union + tn / background_union)


def fuse_rotations(image, predict, rotate):
    """Apply the complete predictor per view, then average aligned raw margins."""
    aligned = []
    for turns in QUARTER_TURNS:
        view = rotate(image, turns, (-2, -1))
        margin = predict(view)
        aligned.append(rotate(margin, -turns, (-2, -1)))
    return sum(aligned) / len(aligned)


def div(a,b):return float(a/b) if b else None


def geom(g):
 from scipy.ndimage import distance_transform_edt
 from skimage.morphology import skeletonize
 sk=skeletonize(g,method='zhang');inside=distance_transform_edt(g)
 outside=distance_transform_edt(~g) if g.any() else np.full(g.shape,np.inf)
 return dict(sk=sk,thin=sk&(inside<=2),far=(~g)&(outside>5),boundary=(g&(inside<=2))|((~g)&(outside<=2)))


def counts(p,g,e):
 from scipy.ndimage import distance_transform_edt, label
 from skimage.morphology import skeletonize
 ps=skeletonize(p,method='zhang');tol=(distance_transform_edt(~p)<=2) if p.any() else np.zeros_like(p)
 gl,ng=label(g,np.ones((3,3)));pl,npred=label(p,np.ones((3,3)))
 ids=e['sk']&(pl>0);coverage=0
 if ids.any():
  keys=gl[ids].astype(np.int64)*(npred+1)+pl[ids];pairs,n=np.unique(keys,return_counts=True);best=np.zeros(ng+1,np.int64);np.maximum.at(best,pairs//(npred+1),n);coverage=int(best[1:].sum())
 return dict(tp=int((p&g).sum()),fp=int((p&~g).sum()),fn=int((~p&g).sum()),tn=int((~p&~g).sum()),thin_hits=int((p&e['thin']).sum()),thin_total=int(e['thin'].sum()),thin_tolerant_hits=int((tol&e['thin']).sum()),far_fp=int((p&e['far']).sum()),far_total=int(e['far'].sum()),pred_skeleton_hits=int((ps&g).sum()),pred_skeleton_total=int(ps.sum()),gt_skeleton_hits=int((e['sk']&p).sum()),gt_skeleton_total=int(e['sk'].sum()),single_component_covered=coverage)


def derive(c):
 precision=div(c['tp'],c['tp']+c['fp']);recall=div(c['tp'],c['tp']+c['fn']);sp=div(c['pred_skeleton_hits'],c['pred_skeleton_total']);sr=div(c['gt_skeleton_hits'],c['gt_skeleton_total'])
 a=div(c['tp'],c['tp']+c['fp']+c['fn']);b=div(c['tn'],c['tn']+c['fp']+c['fn'])
 return dict(global_miou=(a+b)/2 if a is not None and b is not None else None,crack_iou=a,dice=div(2*c['tp'],2*c['tp']+c['fp']+c['fn']),precision=precision,recall=recall,thin_recall=div(c['thin_hits'],c['thin_total']),thin_recall_tolerance2=div(c['thin_tolerant_hits'],c['thin_total']),far_fpr=div(c['far_fp'],c['far_total']),skeleton_precision=sp,skeleton_recall=sr,cldice=2*sp*sr/(sp+sr) if sp is not None and sr is not None and sp+sr>0 else (0. if sp==sr==0 else None),single_component_coverage=div(c['single_component_covered'],c['gt_skeleton_total']))


def configure_native_paths():
    native = str(ROOT / "third_party/cuda")
    if native not in sys.path:
        sys.path.insert(0, native)

def dataset_config(key):
    config = read(ROOT / f"configs/{key}.json")
    return {**read(ROOT / "configs/common.json"), **config}


def split_paths(config, data_root, split):
    rows = read(ROOT / config["split_file"])[split]
    if len(rows) != config["counts"][split]:
        raise ValueError(f"Incorrect number of samples for {split}")
    directory = Path(data_root) / split
    return [(directory / "images" / image, directory / "masks" / mask) for image, mask in rows]


def read_image(path):
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def preprocess_image(image):
    import cv2

    return cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)


def sample(paths):
    import cv2

    image_path, mask_path = paths
    image = read_image(image_path)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(mask_path)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.dtype != np.uint8 or mask.shape != image.shape[:2]:
        raise ValueError(f"Expected a matching uint8 mask for {image_path}")
    mask = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
    return preprocess_image(image), (mask > 127).astype(np.uint8)


def augment(image, mask, epoch, index, train_count):
    from mmseg.datasets.pipelines import Compose

    global _augment
    if _augment is None:
        _augment = Compose([
            dict(type="RandomFlip", prob=.5, direction="horizontal"),
            dict(type="PhotoMetricDistortion"),
            dict(type="Normalize", mean=RGB_MEAN, std=RGB_STD, to_rgb=True),
        ])
    previous = np.random.get_state()
    np.random.seed(4200000 + epoch * train_count + index)
    try:
        result = _augment(dict(img=image.copy(), gt_semantic_seg=mask.copy(),
                               seg_fields=["gt_semantic_seg"], img_fields=["img"]))
        return result["img"].transpose(2, 0, 1).copy(), result["gt_semantic_seg"].copy()
    finally:
        np.random.set_state(previous)


def cuda_device():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ABH requires CUDA. See the repository setup instructions.")
    return torch.device("cuda:0")


def load_body(config, path, device):
    import torch

    configure_native_paths()
    from abh_model import build

    model = build(config)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.reset(checkpoint["step"])
    return model.to(device).eval().requires_grad_(False)


def new_drc(config, device):
    import torch
    from drc_head import Head

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config["drc_seed"])
        head = Head()
    if sum(parameter.numel() for parameter in head.parameters()) != 392:
        raise RuntimeError("Unexpected DRC parameter count")
    return head.to(device)


def body_margin(body, images):
    logits = body.logits(images)
    return logits[:, 1:2] - logits[:, 0:1]


def learning_rate_factor(update_index, horizon, warmup):
    warm = 1e-6 + (1 - 1e-6) * update_index / warmup if update_index < warmup else 1.
    return (1 - update_index / horizon) * warm
