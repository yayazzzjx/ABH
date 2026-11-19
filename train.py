"""Train ABH, export frozen ABH scores, and train DRC."""
import argparse
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import runtime_helpers as rt

def single_view(body, image, device):
    import torch

    normalized = torch.from_numpy(rt.normalize(image))[None].to(device)
    return rt.body_margin(body, normalized)[0, 0].cpu().numpy()


def training_hooks(model, stack):
    from third_party.cuda.training_ops import (
        ComposedDispatchHook, FastR01PathRepairHook, FastR05PathRepairHook,
    )

    repairs = [FastR01PathRepairHook(), FastR05PathRepairHook()]
    composed = ComposedDispatchHook()
    runner = SimpleNamespace(model=model)
    for hook in [*repairs, composed]:
        hook.before_run(runner)
        stack.callback(hook.after_run, runner)
    return repairs, composed


def train_abh(args, config, device):
    import torch

    from abh_model import build, parameter_groups
    from third_party.cuda.training_ops import BICUBIC_MODULES

    train = [rt.sample(pair) for pair in rt.split_paths(config, args.data_root, "train")]
    validation = [rt.sample(pair) for pair in rt.split_paths(config, args.data_root, "val")]
    orders = rt.read(rt.ROOT / config["order_file"])
    horizon = config["abh_epochs"] * config["steps_per_epoch"]
    args.output.mkdir(parents=True, exist_ok=False)
    rt.seed()
    body = build(config, encoder_weights=args.encoder_weights, initial_core=args.initial_core).to(device)
    optimizer = torch.optim.AdamW(parameter_groups(body), betas=(.9, .999), eps=1e-8)
    rt.seed()
    step, best, history = 0, -np.inf, []
    with ExitStack() as cleanup:
        repairs, composed = training_hooks(body, cleanup)
        for epoch, order in enumerate(orders[:config["abh_epochs"]], 1):
            body.train()
            for offset in range(0, len(order), 2):
                items = []
                for position, index in enumerate(order[offset:offset + 2], offset):
                    augmentation_index = index
                    if len(train) % 2 and position == len(order) - 1:
                        augmentation_index += len(train) * config["abh_epochs"]
                    items.append(rt.augment(*train[index], epoch, augmentation_index, len(train)))
                images, labels = (torch.from_numpy(np.stack(values)).to(device) for values in zip(*items))
                factor = rt.learning_rate_factor(step, horizon, config["warmup_iters"])
                for group in optimizer.param_groups:
                    group["lr"] = group["base_lr"] * factor
                optimizer.zero_grad(set_to_none=True)
                composed.calls.clear()
                for hook in repairs:
                    hook.calls.clear()
                step += 1
                loss = sum(body.loss(images, labels[:, None], step).values())
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite ABH loss at step {step}")
                loss.backward()
                if ([row["module"] for row in composed.calls] != list(BICUBIC_MODULES)
                        or not all(row["backward_seen"] for row in composed.calls)
                        or sorted(repairs[1].calls) != ["POOL", "REAL16", "REAL32", "S1"]):
                    raise RuntimeError("ABH training interpolation dispatch changed")
                optimizer.step()

            body.eval()
            state = rt.rng()
            try:
                torch.manual_seed(424242)
                torch.cuda.manual_seed_all(424242)
                with torch.inference_mode():
                    metrics = rt.select_on_validation(
                        (single_view(body, image, device), label) for image, label in validation)
            finally:
                rt.restore(state)
            row = {"epoch": epoch, "step": step, **metrics}
            if row["mIoU"] > best:
                best = row["mIoU"]
                rt.save(args.output / "best_abh.pth", {"model": body.state_dict(), "step": step})
            history.append(row)
            rt.write(args.output / "history.json", history)
            print(row, flush=True)


def export_cache(args, config, device):
    import torch

    args.output.mkdir(parents=True, exist_ok=False)
    body = rt.load_body(config, args.abh_weights, device)
    before = rt.statehash(body)
    with torch.inference_mode():
        for split in ("train", "val"):
            paths = rt.split_paths(config, args.data_root, split)
            shape = (len(paths), *rt.IMAGE_SIZE)
            bgr = np.lib.format.open_memmap(args.output / f"{split}_bgr.npy", "w+", np.uint8, (*shape, 3))
            masks = np.lib.format.open_memmap(args.output / f"{split}_masks.npy", "w+", np.uint8, shape)
            margins = np.lib.format.open_memmap(args.output / f"{split}_margins.npy", "w+", np.float32, shape)
            for index, pair in enumerate(paths):
                image, mask = rt.sample(pair)
                margin = single_view(body, image, device)
                if not np.isfinite(margin).all():
                    raise RuntimeError(f"Non-finite ABH score for {pair[0]}")
                bgr[index], masks[index], margins[index] = image, mask, margin
            for array in (bgr, masks, margins):
                array.flush()
            print(f"Cached {split}: {len(paths)} images", flush=True)
    if rt.statehash(body) != before:
        raise RuntimeError("Frozen ABH state changed while exporting the cache")
    rt.write(args.output / "cache.json", {
        "dataset": args.dataset,
        "splits_sha256": rt.sha(rt.ROOT / config["split_file"]),
        "abh_sha256": rt.sha(args.abh_weights),
    })


def cache_split(directory, split, count):
    values = [np.load(directory / f"{split}_{name}.npy", mmap_mode="r")
              for name in ("bgr", "margins", "masks")]
    shape = (count, *rt.IMAGE_SIZE)
    if [array.shape for array in values] != [(*shape, 3), shape, shape]:
        raise ValueError(f"Incomplete {split} cache")
    return values


def cache_batch(data, indices, device):
    import torch

    images, margins, labels = data
    rgb = np.stack([rt.normalize(images[index]) for index in indices])
    scores = np.array(margins[indices])[:, None]
    masks = np.array(labels[indices])[:, None]
    return tuple(torch.from_numpy(value).to(device) for value in (rgb, scores, masks))


def train_drc(args, config, device):
    import torch
    from drc_head import softcl

    provenance = rt.read(args.cache / "cache.json")
    if (provenance["dataset"] != args.dataset
            or provenance["splits_sha256"] != rt.sha(rt.ROOT / config["split_file"])):
        raise ValueError("Cache dataset or split does not match the selected configuration")
    train = cache_split(args.cache, "train", config["counts"]["train"])
    validation = cache_split(args.cache, "val", config["counts"]["val"])
    orders = rt.read(rt.ROOT / config["order_file"])
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    head = rt.new_drc(config, device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config["drc_lr"], weight_decay=.01)
    horizon = config["abh_epochs"] * config["steps_per_epoch"]
    ramp_steps = config["drc_ramp_epochs"] * config["steps_per_epoch"]

    def validate():
        head.eval()
        with torch.inference_mode():
            def predictions():
                for index, label in enumerate(validation[2]):
                    rgb, margin, _ = cache_batch(validation, [index], device)
                    refined = head(rgb, margin)[0, 0].cpu().numpy()
                    yield refined, label
            return rt.select_on_validation(predictions())

    best = validate()
    history = [{"epoch": 0, "step": 0, **best}]
    rt.save(args.output / "best_drc.pth", head.state_dict())
    rt.write(args.output / "cache_source.json", provenance)
    step = 0
    for epoch, order in enumerate(orders[:config["drc_epochs"]], 1):
        head.train()
        for offset in range(0, len(order), 2):
            rgb, margin, mask = cache_batch(train, order[offset:offset + 2], device)
            factor = rt.learning_rate_factor(step, horizon, config["warmup_iters"])
            optimizer.param_groups[0]["lr"] = config["drc_lr"] * factor
            step += 1
            optimizer.zero_grad(set_to_none=True)
            weight = config["drc_loss_weight"] * min(1., step / ramp_steps)
            loss = weight * softcl(head(rgb, margin), mask)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite DRC loss at step {step}")
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                head.amplitude.clamp_(0, 1)
        metrics = validate()
        if metrics["mIoU"] > best["mIoU"]:
            best = metrics
            rt.save(args.output / "best_drc.pth", head.state_dict())
        row = {"epoch": epoch, "step": step, **metrics}
        history.append(row)
        rt.write(args.output / "history.json", history)
        print(row, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    abh = commands.add_parser("abh", help="Train the ABH body")
    cache = commands.add_parser("cache", help="Export train and val scores from a frozen ABH")
    drc = commands.add_parser("drc", help="Train the DRC head on the frozen cache")
    for command in (abh, cache, drc):
        command.add_argument("--dataset", choices=rt.DATASETS, required=True)
        command.add_argument("--output", type=Path, required=True)
    for command in (abh, cache):
        command.add_argument("--data-root", type=Path, required=True)
    abh.add_argument("--encoder-weights", type=Path, required=True)
    abh.add_argument("--initial-core", type=Path, help="Optional original zero-update core")
    cache.add_argument("--abh-weights", type=Path, required=True)
    drc.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    rt.seed()
    rt.configure_native_paths()
    config = rt.dataset_config(args.dataset)
    {"abh": train_abh, "cache": export_cache, "drc": train_drc}[args.stage](args, config, rt.cuda_device())


if __name__ == "__main__":
    main()
