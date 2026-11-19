"""Run ABH and DRC with four-view rotation and validation-selected thresholds."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

import runtime_helpers as rt


def predict(image, body, drc, device):
    import torch

    images = torch.from_numpy(rt.normalize(image))[None].to(device)
    with torch.inference_mode():
        fused = rt.fuse_rotations(
            images, lambda view: drc(view, rt.body_margin(body, view)), torch.rot90)
    margin = fused[0, 0].cpu().numpy()
    if not np.isfinite(margin).all():
        raise RuntimeError("Model returned a non-finite margin")
    return margin


def model_identity(args, config):
    return {
        "protocol": rt.PROTOCOL,
        "dataset": args.dataset,
        "split_sha256": rt.sha(rt.ROOT / config["split_file"]),
        "abh_sha256": rt.sha(args.abh_weights),
        "drc_sha256": rt.sha(args.drc_weights),
        "rotation_degrees_ccw": [90 * turns for turns in rt.QUARTER_TURNS],
    }


def read_lock(path, identity):
    lock = rt.read(path)
    if any(lock.get(key) != value for key, value in identity.items()):
        raise ValueError("Validation lock does not match the model, dataset or protocol")
    threshold = lock["validation"]["threshold"]
    if threshold not in np.arange(0., 1., .01):
        raise ValueError("Validation threshold is outside the protocol grid")
    return lock, threshold


def load_models(args, config, device):
    import torch

    body = rt.load_body(config, args.abh_weights, device)
    drc = rt.new_drc(config, device)
    drc.load_state_dict(torch.load(args.drc_weights, map_location="cpu", weights_only=True), strict=True)
    return body, drc.eval().requires_grad_(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    infer = commands.add_parser("predict", help="Save a 512 x 512 raw margin from one image")
    val = commands.add_parser("validate", help="Select and save a threshold using the validation split")
    test = commands.add_parser("test", help="Evaluate the test split with a saved validation threshold")
    for command in (infer, val, test):
        command.add_argument("--dataset", choices=rt.DATASETS, required=True)
        command.add_argument("--abh-weights", type=Path, required=True)
        command.add_argument("--drc-weights", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    infer.add_argument("--image", type=Path, required=True)
    test.add_argument("--validation-lock", type=Path, required=True)
    for command in (val, test):
        command.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    config = rt.dataset_config(args.dataset)
    if args.mode == "test" and args.output.resolve() == args.validation_lock.resolve():
        parser.error("Choose a different --output from --validation-lock")
    identity = model_identity(args, config)
    if args.mode == "test":
        lock, threshold = read_lock(args.validation_lock, identity)
    rt.seed()
    device = rt.cuda_device()
    body, drc = load_models(args, config, device)
    if args.mode == "predict":
        image = rt.preprocess_image(rt.read_image(args.image))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("wb") as file:
            np.save(file, predict(image, body, drc, device), allow_pickle=False)
        return

    def margins(split):
        for pair in rt.split_paths(config, args.data_root, split):
            image, mask = rt.sample(pair)
            yield predict(image, body, drc, device), mask

    if args.mode == "validate":
        validation = rt.select_on_validation(margins("val"))
        rt.write(args.output, {**identity, "validation": validation})
        return

    above_threshold = np.arange(256, dtype=np.float64) / 255 > threshold
    test_scores = []
    structural_counts = Counter()
    for margin, mask in margins("test"):
        target = mask.astype(bool)
        test_scores.append(rt.image_miou(above_threshold[rt.export(margin)], target))
        structural_counts.update(rt.counts(margin > 0, target, rt.geom(target)))
    lock["test"] = {
        "samples": len(test_scores),
        "mIoU": float(np.mean(test_scores)),
        "threshold_from_validation": threshold,
        "structural_raw_margin_threshold": 0.0,
        "structural_counts": dict(structural_counts),
        "structural_metrics": rt.derive(structural_counts),
    }
    rt.write(args.output, lock)


if __name__ == "__main__":
    main()
