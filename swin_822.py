from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    CropForegroundd,
    SpatialPadd,
    ScaleIntensityRangePercentilesd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotated,
    RandAffined,
    Rand3DElasticd,
    RandGridDistortiond,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandShiftIntensityd,
    RandScaleIntensityd,
    RandZoomd,
    RandHistogramShiftd,
    KeepLargestConnectedComponent,
)
from monai.networks.nets import SwinUNETR
from monai.metrics import DiceMetric
from monai.losses import DiceFocalLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, MetaTensor, decollate_batch, list_data_collate
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
import importlib
import math
import os
import random
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")


def ensure_packages(packages):
    missing = []
    for pkg in packages:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)


ensure_packages(["monai", "einops", "nibabel"])


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass
try:
    torch.set_float32_matmul_precision("medium")
except AttributeError:
    pass

CONFIG = {
    "images_dir": "/kaggle/input/imagess/volumes/",
    "labels_dir": "/kaggle/input/labels/masks/",
    "best_model_path": "/kaggle/working/best_swin_fast.pth",
    "seed": 42,
    "epochs": 120,
    "batch_size": 1,
    "accum_steps": 2,
    "val_every": 2,
    "base_lr": 2e-4,
    "min_lr": 5e-6,
    "weight_decay": 1e-5,
    "warmup_epochs": 5,
    "max_grad_norm": 1.0,
    "feature_size": 16,
    "drop_rate": 0.0,
    "window_size": 6,
    "target_spacing": (1.0, 1.0, 0.5),  # depth shouldn't be more than 48
    "roi_size": (96, 96, 96),  # needs divisible by 32
    "crop_margin": 20,
    "cache_rate": 0.85,  # for speed
    "train_cache_workers": 2,
    "val_cache_workers": 2,
    "swi_batch_size": 1,
    "overlap": 0.4,
    "init_threshold": 0.5,
    "patience": 25,
    "min_delta": 0.001,
    "num_samples": 1,
    "test_size": 0.2,
    "num_workers": 4,
    "val_num_workers": 2,
    "prefetch_factor": 2,
    "val_prefetch_factor": 2,
    "persistent_workers": False,
    "val_persistent_workers": False,
    "use_tta": False,
    "tta_flips": [(), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)],
    "use_ema": True,
    "ema_decay": 0.995,
    "use_compile": False,
    "thr_sweep_every": 10,
    "thr_grid": [0.35, 0.45, 0.55, 0.65, 0.75],
}

Path(CONFIG["best_model_path"]).parent.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NGPU = torch.cuda.device_count()
print(f"CUDA: {torch.cuda.get_device_name(0) if NGPU else 'CPU'} | GPUs={NGPU}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = CONFIG["seed"] + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)


seed_everything(CONFIG["seed"])

post_pred = KeepLargestConnectedComponent(applied_labels=[1], connectivity=3)


def get_transforms():
    base_transform = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=CONFIG["target_spacing"], mode=("bilinear", "nearest")),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=1.0, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="label", margin=CONFIG["crop_margin"]),
        SpatialPadd(keys=["image", "label"], spatial_size=CONFIG["roi_size"], method="symmetric"),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32, track_meta=False),
    ])

    train_transform = Compose([
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=CONFIG["roi_size"],
            pos=1,
            neg=1,
            num_samples=CONFIG["num_samples"],
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotated(
            keys=["image", "label"],
            prob=0.2,
            range_x=0.25,
            range_y=0.25,
            range_z=0.25,
            mode=("bilinear", "nearest"),
        ),
        RandAffined(
            keys=["image", "label"],
            prob=0.15,
            translate_range=(5, 5, 5),
            rotate_range=(0, 0, math.pi / 12),
            scale_range=(0.1, 0.1, 0.1),
            mode=("bilinear", "nearest"),
            padding_mode="border",
        ),
        RandGridDistortiond(
            keys=["image", "label"],
            prob=0.1,
            distort_limit=(-0.015, 0.015),
            mode=("bilinear", "nearest"),
        ),
        RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.05),
        RandGaussianSmoothd(
            keys=["image"],
            prob=0.1,
            sigma_x=(0.5, 1.5),
            sigma_y=(0.5, 1.5),
            sigma_z=(0.5, 1.5),
        ),
        RandAdjustContrastd(keys=["image"], prob=0.15, gamma=(0.8, 1.4)),
        RandHistogramShiftd(keys=["image"], prob=0.1, num_control_points=(5, 12)),
        RandShiftIntensityd(keys=["image"], prob=0.2, offsets=0.1),
        RandScaleIntensityd(keys=["image"], prob=0.2, factors=0.1),
        RandZoomd(
            keys=["image", "label"],
            prob=0.15,
            min_zoom=0.92,
            max_zoom=1.08,
            mode=("trilinear", "nearest"),
        ),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32, track_meta=False),
    ])

    val_transform = Compose([
        EnsureTyped(keys=["image", "label"], dtype=torch.float32, track_meta=False),
    ])

    return base_transform, train_transform, val_transform


class PlacentaDataset(Dataset):
    def __init__(self, images_dir, labels_dir, limit=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        image_files = sorted(self.images_dir.glob("*.nii*"))
        label_files = sorted(self.labels_dir.glob("*.nii*"))
        label_map = {
            f.name.replace("_mask", "").replace(".nii", "").replace(".gz", ""): f for f in label_files
        }
        self.pairs = []
        for img_path in image_files:
            stem = img_path.name.replace(".nii", "").replace(".gz", "")
            lbl_path = label_map.get(stem)
            if lbl_path and lbl_path.exists():
                self.pairs.append({"image": str(img_path), "label": str(lbl_path)})
        if limit:
            self.pairs = self.pairs[:limit]
        print(f"[Dataset] Found {len(self.pairs)} pairs.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return self.pairs[index]


def safe_collate(batch):
    def to_tensor(obj):
        if isinstance(obj, MetaTensor):
            # Strip MONAI metadata so default torch collation is used
            return obj.as_tensor()
        if isinstance(obj, np.ndarray):
            return torch.as_tensor(obj)
        if isinstance(obj, dict):
            return {k: to_tensor(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_tensor(v) for v in obj]
        return obj

    converted = [to_tensor(sample) for sample in batch]
    return list_data_collate(converted)


def build_loader(dataset, batch_size, shuffle, num_workers, prefetch_factor, persistent_workers):
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": safe_collate,
        "worker_init_fn": seed_worker,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers
    return DataLoader(dataset, **loader_kwargs)


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.ema_model = copy.deepcopy(model).to(device)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        ema_state = self.ema_model.state_dict()
        model_state = model.state_dict()
        for key, value in model_state.items():
            if value.dtype.is_floating_point:
                ema_state[key].mul_(self.decay).add_(value, alpha=1.0 - self.decay)
            else:
                ema_state[key].copy_(value)

    def state_dict(self):
        return self.ema_model.state_dict()


def sliding_window_predict(inputs, model, amp_enabled: bool):
    if CONFIG["use_tta"]:
        predictions = []
        for dims in CONFIG["tta_flips"]:
            if dims:
                aug_inputs = torch.flip(inputs, dims)
            else:
                aug_inputs = inputs
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = sliding_window_inference(
                    aug_inputs,
                    CONFIG["roi_size"],
                    CONFIG["swi_batch_size"],
                    model,
                    overlap=CONFIG["overlap"],
                    mode="gaussian",
                )
            if dims:
                logits = torch.flip(logits, dims)
            predictions.append(logits)
        return torch.mean(torch.stack(predictions, dim=0), dim=0)

    with torch.cuda.amp.autocast(enabled=amp_enabled):
        return sliding_window_inference(
            inputs,
            CONFIG["roi_size"],
            CONFIG["swi_batch_size"],
            model,
            overlap=CONFIG["overlap"],
            mode="gaussian",
        )


@torch.no_grad()
def validate(model, loader, thresholds, amp_enabled: bool):
    model.eval()
    thr_list = thresholds if isinstance(thresholds, (list, tuple)) else [thresholds]
    dice_metrics = {thr: DiceMetric(include_background=False, reduction="mean") for thr in thr_list}

    for batch in tqdm(loader, desc="Val", leave=False):
        val_inputs = batch["image"].to(device)
        val_labels = decollate_batch(batch["label"].to(device))
        probs = torch.sigmoid(sliding_window_predict(val_inputs, model, amp_enabled))
        prob_maps = decollate_batch(probs)
        for thr, metric in dice_metrics.items():
            thr_outputs = [(p > thr).float() for p in prob_maps]
            thr_outputs = [post_pred(i) for i in thr_outputs]
            metric(y_pred=thr_outputs, y=val_labels)

    scores = {thr: metric.aggregate().item() for thr, metric in dice_metrics.items()}
    for metric in dice_metrics.values():
        metric.reset()
    return scores


def warmup_cosine_lambda(epoch: int) -> float:
    if epoch < CONFIG["warmup_epochs"]:
        return float(epoch + 1) / max(1, CONFIG["warmup_epochs"])
    progress = (epoch - CONFIG["warmup_epochs"]) / max(1, CONFIG["epochs"] - CONFIG["warmup_epochs"])
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max(cosine, CONFIG["min_lr"] / CONFIG["base_lr"])


def main():
    base_tfm, train_tfm, val_tfm = get_transforms()

    full_ds = PlacentaDataset(CONFIG["images_dir"], CONFIG["labels_dir"])
    indices = np.arange(len(full_ds))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["seed"],
        shuffle=True,
    )
    train_files = [full_ds[i] for i in train_idx]
    val_files = [full_ds[i] for i in val_idx]
    print(f"[Split] Train={len(train_files)} | Val={len(val_files)}")

    print("[Cache] Priming training volumes in RAM...")
    train_cache = CacheDataset(
        data=train_files,
        transform=base_tfm,
        cache_rate=CONFIG["cache_rate"],
        num_workers=CONFIG["train_cache_workers"],
    )
    val_cache = CacheDataset(
        data=val_files,
        transform=base_tfm,
        cache_rate=CONFIG["cache_rate"],
        num_workers=CONFIG["val_cache_workers"],
    )

    train_ds = Dataset(train_cache, transform=train_tfm)
    val_ds = Dataset(val_cache, transform=val_tfm)

    train_loader = build_loader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        prefetch_factor=CONFIG["prefetch_factor"],
        persistent_workers=CONFIG["persistent_workers"],
    )
    val_loader = build_loader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=CONFIG["val_num_workers"],
        prefetch_factor=CONFIG["val_prefetch_factor"],
        persistent_workers=CONFIG["val_persistent_workers"],
    )

    model = SwinUNETR(
        in_channels=1,
        out_channels=1,
        feature_size=CONFIG["feature_size"],
        patch_size=2,
        window_size=CONFIG["window_size"],
        spatial_dims=3,
        drop_rate=CONFIG["drop_rate"],
        attn_drop_rate=CONFIG["drop_rate"],
        depths=(2, 2, 2, 2),
        use_checkpoint=True,
    ).to(device)

    if CONFIG["use_compile"] and NGPU <= 1:
        try:
            print("[Speedup] Compiling model with torch.compile...")
            model = torch.compile(model)
        except Exception as compile_err:
            print(f"[Warning] torch.compile skipped: {compile_err}")

    if NGPU > 1:
        print(f"[Multi-GPU] Using DataParallel on {NGPU} GPUs")
        model = torch.nn.DataParallel(model)

    backbone = model.module if isinstance(model, torch.nn.DataParallel) else model
    ema_helper = ModelEMA(backbone, CONFIG["ema_decay"]) if CONFIG["use_ema"] else None

    criterion = DiceFocalLoss(include_background=False, sigmoid=True, lambda_dice=1.0, lambda_focal=1.0, gamma=2.0)
    optimizer = torch.optim.AdamW(backbone.parameters(), lr=CONFIG["base_lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    amp_enabled = scaler.is_enabled()

    history = {"train_loss": [], "val_dice": [], "val_epochs": []}
    best_dice = 0.0
    best_epoch = -1
    best_threshold = CONFIG["init_threshold"]
    best_threshold_for_best = best_threshold
    no_improve = 0

    print(f"Starting training for {CONFIG['epochs']} epochs...")
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_loss = 0.0
        num_steps = 0
        accum_counter = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']}", leave=False)

        for batch in pbar:
            inputs = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            loss_value = loss.item()
            epoch_loss += loss_value
            num_steps += 1

            scaler.scale(loss / CONFIG["accum_steps"]).backward()
            accum_counter += 1

            if accum_counter == CONFIG["accum_steps"]:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema_helper:
                    ema_helper.update(backbone)
                accum_counter = 0

            pbar.set_postfix({"loss": f"{loss_value:.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

        if accum_counter > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema_helper:
                ema_helper.update(backbone)

        avg_loss = epoch_loss / max(num_steps, 1)
        history["train_loss"].append(avg_loss)

        if (epoch + 1) % CONFIG["val_every"] == 0:
            eval_model = ema_helper.ema_model if ema_helper else backbone
            sweep = (epoch + 1) % CONFIG["thr_sweep_every"] == 0
            thresholds = CONFIG["thr_grid"] if sweep else [best_threshold]
            val_scores = validate(eval_model, val_loader, thresholds=thresholds, amp_enabled=amp_enabled)
            current_thr = max(val_scores, key=val_scores.get)
            val_dice = val_scores[current_thr]
            best_threshold = current_thr
            history["val_dice"].append(val_dice)
            history["val_epochs"].append(epoch + 1)

            print(
                f"Epoch {epoch + 1} | Loss: {avg_loss:.4f} | Val Dice: {val_dice:.4f} @thr={current_thr:.2f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

            if val_dice > best_dice + CONFIG["min_delta"]:
                best_dice = val_dice
                best_epoch = epoch + 1
                best_threshold_for_best = current_thr
                no_improve = 0
                torch.save((ema_helper.ema_model if ema_helper else backbone).state_dict(), CONFIG["best_model_path"])
                print(f"--> New Best Model! Dice={best_dice:.4f} @thr={current_thr:.2f}")
            else:
                no_improve += 1

            if no_improve >= CONFIG["patience"]:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        else:
            print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f}")

        scheduler.step()

    print(f"Training Finished. Best Dice: {best_dice:.4f} at Epoch {best_epoch} (thr={best_threshold_for_best:.2f})")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["val_epochs"], history["val_dice"], label="Val Dice", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Validation Dice")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_plot.png")
    print("Plot saved to training_plot.png")


if __name__ == "__main__":
    main()
