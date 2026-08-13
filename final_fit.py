"""Train PGCC-Net for a fixed number of epochs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import Model


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "Data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def validate_inputs(data_dir: Path) -> dict[str, Path]:
    paths = {
        "images": data_dir / "Combined_Images.npy",
        "angles": data_dir / "Combined_Angles.npy",
        "metadata": data_dir / "Combined_Metadata.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required data files are missing:\n  - " + "\n  - ".join(missing)
        )
    return paths


def configure(paths: dict[str, Path], output_dir: Path, args: argparse.Namespace) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = Model.CONFIG
    config.update(
        {
            "IMG_PATH": str(paths["images"]),
            "ANG_PATH": str(paths["angles"]),
            "LBL_PATH": str(paths["metadata"]),
            "TRAIN_INDEX_PATH": str(output_dir / "train_indices.npy"),
            "STATS_PATH": str(output_dir / "stats.json"),
            "CI_BG_PATH": str(output_dir / "ci_background.npz"),
            "FEATURE_CACHE_PREFIX": str(output_dir / "feature_cache"),
            "EPOCHS": int(args.epochs),
            "BATCH_SIZE": int(args.batch_size),
            "NUM_WORKERS": int(args.num_workers),
        }
    )
    return config


def validate_training_indices(metadata, indices: np.ndarray) -> None:
    station_ids = Model.infer_station_ids_from_df(metadata, len(metadata)).astype(str)
    selected = set(np.unique(station_ids[indices]).tolist())
    expected = set(Model.TRAINING_STATIONS)
    if selected != expected:
        raise ValueError(
            "Training station mismatch. Missing: {}; unexpected: {}".format(
                sorted(expected - selected), sorted(selected - expected)
            )
        )
    if expected.intersection(Model.SPATIAL_TEST_STATIONS):
        raise ValueError("Training stations and spatial test stations overlap")


def make_loader(dataset, indices: np.ndarray, config: dict) -> DataLoader:
    generator = torch.Generator().manual_seed(int(config["RANDOM_SEED"]))
    kwargs = {
        "batch_size": int(config["BATCH_SIZE"]),
        "shuffle": True,
        "drop_last": False,
        "num_workers": int(config["NUM_WORKERS"]),
        "pin_memory": torch.cuda.is_available(),
        "generator": generator,
    }
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(config["PREFETCH_FACTOR"])
    return DataLoader(Subset(dataset, indices.tolist()), **kwargs)


def train_one_epoch(model, criterion, loader, optimizer, scaler, use_amp, channels_last):
    model.train()
    keys = ["loss", "loss_g", "loss_dir", "loss_dif", "loss_phy"]
    totals = {key: 0.0 for key in keys}
    sample_count = 0

    progress = tqdm(loader, leave=False)
    for batch in progress:
        batch_size = int(batch["y"].shape[0])
        if channels_last:
            images = batch["x_img"].to(
                Model.DEVICE,
                non_blocking=True,
                memory_format=torch.channels_last,
            )
        else:
            images = batch["x_img"].to(Model.DEVICE, non_blocking=True)
        angles = batch["x_ang"].to(Model.DEVICE, non_blocking=True)
        physical = batch["x_phys"].to(Model.DEVICE, non_blocking=True)
        targets = batch["y"].to(Model.DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred_g, pred_dir, pred_dif, _ = model(images, angles, physical)
            loss, logs = criterion(
                pred_g,
                pred_dir,
                pred_dif,
                targets[:, 0:1],
                targets[:, 1:2],
                targets[:, 2:3],
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(criterion.parameters()), max_norm=5.0
        )
        scaler.step(optimizer)
        scaler.update()

        totals["loss"] += float(loss.detach().cpu()) * batch_size
        for key in keys[1:]:
            totals[key] += float(logs[key]) * batch_size
        sample_count += batch_size
        progress.set_postfix(loss="{:.4f}".format(float(loss.detach().cpu())))

    return {key: value / max(1, sample_count) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    paths = validate_inputs(args.data_dir.resolve())
    output_dir = args.output_dir.resolve()
    config = configure(paths, output_dir, args)
    Model.set_seed(int(config["RANDOM_SEED"]))

    metadata = Model.read_model_metadata(config["LBL_PATH"])
    train_indices = Model.station_indices(metadata, Model.TRAINING_STATIONS)
    validate_training_indices(metadata, train_indices)
    np.save(config["TRAIN_INDEX_PATH"], train_indices)

    dataset = Model.NativeDataset(
        config["IMG_PATH"],
        config["ANG_PATH"],
        config["LBL_PATH"],
        config["STATS_PATH"],
        metadata=metadata,
    )
    loader = make_loader(dataset, train_indices, config)

    model = Model.make_model().to(Model.DEVICE)
    channels_last = bool(config["CHANNELS_LAST"]) and torch.cuda.is_available()
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    criterion = Model.AdaptiveMSESoftClosureLoss(
        config["STATS_PATH"], config["ADAPTIVE_LOG_VARS_INIT"]
    ).to(Model.DEVICE)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=float(config["LEARNING_RATE"]),
    )
    use_amp = bool(config["USE_AMP"]) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    run_config = {
        "epochs": int(config["EPOCHS"]),
        "batch_size": int(config["BATCH_SIZE"]),
        "training_samples": int(len(train_indices)),
        "training_stations": list(Model.TRAINING_STATIONS),
        "spatial_test_stations_used": False,
        "device": str(Model.DEVICE),
        "trainable_parameters": Model.count_parameters(model),
        "config": config,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    print("Device: {}".format(Model.DEVICE))
    print("Training samples: {:,}".format(len(train_indices)))
    print("Training epochs: {}".format(config["EPOCHS"]))
    print("Trainable parameters: {:,}".format(Model.count_parameters(model)))

    history = []
    for epoch in range(int(config["EPOCHS"])):
        metrics = train_one_epoch(
            model, criterion, loader, optimizer, scaler, use_amp, channels_last
        )
        row = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **metrics,
            "weight_g": float(torch.exp(-criterion.log_vars[0]).detach().cpu()),
            "weight_dir": float(torch.exp(-criterion.log_vars[1]).detach().cpu()),
            "weight_dif": float(torch.exp(-criterion.log_vars[2]).detach().cpu()),
            "weight_phy": float(torch.exp(-criterion.log_vars[3]).detach().cpu()),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "criterion_state_dict": criterion.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "config": config,
                "training_stations": list(Model.TRAINING_STATIONS),
                "train_index_fingerprint": Model.index_fingerprint(train_indices),
                "history": history,
            },
            output_dir / "last_checkpoint.pth",
        )
        print(
            "Epoch {}/{}: loss={:.5f}, GHI={:.5f}, Direct={:.5f}, "
            "Diffuse={:.5f}, Closure={:.5f}".format(
                epoch + 1,
                config["EPOCHS"],
                row["loss"],
                row["loss_g"],
                row["loss_dir"],
                row["loss_dif"],
                row["loss_phy"],
            )
        )

    final_path = output_dir / "pgcc_net.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "config": config,
            "epochs": int(config["EPOCHS"]),
            "training_stations": list(Model.TRAINING_STATIONS),
            "train_index_fingerprint": Model.index_fingerprint(train_indices),
            "history": history,
        },
        final_path,
    )
    print("Training complete: {}".format(final_path))


if __name__ == "__main__":
    main()
