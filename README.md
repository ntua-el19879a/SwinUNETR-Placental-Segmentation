<<<<<<< HEAD
# SwinUNETR-Placental-Segmentation
My Diploma's Thesis  project with the task of training SwinUNETR to segment Placentas in MRI images.
=======
# SwinUNETR Placenta Segmentation

`swin_822.py` trains a 3D SwinUNETR model (MONAI) for placenta segmentation from NIfTI volumes. It is set up for Kaggle-style workflows and includes caching, strong 3D data augmentation, mixed precision, gradient accumulation, EMA, validation threshold sweeps, early stopping, and plotting of training curves.

## Data layout
- Update `CONFIG["images_dir"]` and `CONFIG["labels_dir"]` to point to folders of `.nii`/`.nii.gz` volumes and masks. A mask named `case_mask.nii.gz` pairs with an image named `case.nii.gz`.
- The best checkpoint is written to `CONFIG["best_model_path"]` (defaults to `/kaggle/working/best_swin_fast.pth`), and a loss/validation plot is saved as `training_plot.png`.

## Dependencies
Tested with Python 3.10+ and PyTorch on GPU. Install the core requirements with:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install monai nibabel einops scikit-learn matplotlib tqdm
```

## Running
1. Set any CONFIG values you want to change near the top of `swin_822.py` (paths, batch size, ROI size, learning rates, TTA, etc.).
2. Launch training:
   ```bash
   python swin_822.py
   ```
3. Watch the console for training/validation Dice updates. The best model (optionally EMA-smoothed) is saved automatically.

## Notable training features
- Warmup + cosine LR schedule, Dice+Focal loss, gradient clipping, and accumulation (`accum_steps`).
- Optional TTA during validation, threshold grid search every `thr_sweep_every` epochs, and early stopping with patience/min-delta.
- Caches preprocessed volumes in RAM for faster epochs; uses sliding-window inference for full-volume validation.
>>>>>>> cc35d1c (Add swin_822 training script and README)
