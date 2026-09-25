"""
Loads a saved checkpoint (produced by direct_main_commented.py) and
evaluates the trained model on the validation set, WITHOUT retraining.

Usage:
    python evaluate_model.py my_config.json /path/to/checkpoint_final.pth

Prints per-variable RMSE and correlation (predicted vs. true, in
de-normalized/physical units), and shows example predictions for a
handful of individual samples.
"""

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

import dataloader_direct as dataloader
from direct_models import *   # brings in the model classes, e.g. Direct_w_Resblock_seq


## ---------------------------------------------------------------------
## Parse arguments and load config
## ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description='evaluate a trained Direct model')
parser.add_argument('config_file', type=str)
parser.add_argument('checkpoint_path', type=str)
args = parser.parse_args()

config = json.load(open(args.config_file))
device = config['device']


## ---------------------------------------------------------------------
## Prepare validation dataset (same as during training)
## ---------------------------------------------------------------------
val_dataset = dataloader.direct_dataset(
    config['target_array'], config['spectrum_h5'],
    i_start_ds=config['i_start_val_ds'], i_end_ds=config['i_end_val_ds'],
    mean_normalization_json=config['mean_normalization_json'],
    std_normalization_json=config['std_normalization_json'],
    mean_normalization_npy=config['mean_normalization_npy'],
    std_normalization_npy=config['std_normalization_npy'],
    normalize_spectra=config['normalize_spectra'],
    normalize_target=config['normalize_target']
)

val_dataloader = DataLoader(val_dataset, batch_size=config['val_batch_size'],num_workers=config.get('num_workers_loader', 0),pin_memory=(device != 'cpu'))


## ---------------------------------------------------------------------
## Rebuild the model architecture and load the trained weights
## ---------------------------------------------------------------------
# NOTE: model_name/n_channels must match what the checkpoint was trained
# with -- these come from the same config file used for training, so as
# long as you pass the matching config this is handled automatically.
model_name = config['model_name']
n_channels = config['n_channels']
model = locals()[model_name](n_channels).to(device)

checkpoint = torch.load(args.checkpoint_path, map_location=device) #load model
model.load_state_dict(checkpoint['model_state'])#load model

# IMPORTANT: eval() switches BatchNorm layers to use their running
# statistics (accumulated during training) instead of per-batch stats --
# the correct mode for inference. Forgetting this is a common source of
# misleadingly bad evaluation results.
model.eval()

print(f"Loaded checkpoint from {args.checkpoint_path} "
      f"({checkpoint.get('nb_epochs_finished', '?')} epochs trained).")


## ---------------------------------------------------------------------
## Run the model over the full validation set
## ---------------------------------------------------------------------
all_preds = []
all_targets = []

with torch.no_grad():
    for input_data, target in val_dataloader:
        input_data = input_data.to(device)
        output = model(input_data)[:, 0, :]   # shape (batch, 13)
        all_preds.append(output.cpu())
        all_targets.append(target)

all_preds = torch.cat(all_preds).numpy()
all_targets = torch.cat(all_targets).numpy()


## ---------------------------------------------------------------------
## De-normalize back to physical units
## ---------------------------------------------------------------------
target_mean = val_dataset.target_mean_for_normalization.numpy()
target_std = val_dataset.target_std_for_normalization.numpy()

pred_denorm = all_preds * target_std + target_mean
target_denorm = all_targets * target_std + target_mean


## ---------------------------------------------------------------------
## Per-variable metrics
## ---------------------------------------------------------------------
# Order must match exactly how target_array's columns were built --
# see the data-generation script (targets, targets_mean, targets_std).
target_keys = ['a_mass_size', 'alpha_area_size', 'aspect_ratio', 'b_mass_size','beta_area_size', 'dmean', 'iwc', 'noise_level_W', 'noise_level_Ka','windW', 'windKa', 'SIGMA_TURB_1', 'SIGMA_TURB_0',
]

print(f"\nEvaluated on {len(target_denorm)} validation samples.\n")
print(f"{'variable':20s} {'RMSE':>10s} {'correlation':>12s}")
print("-" * 44)
for j, key in enumerate(target_keys[:pred_denorm.shape[1]]):
    rmse = np.sqrt(np.mean((pred_denorm[:, j] - target_denorm[:, j]) ** 2))
    corr = np.corrcoef(pred_denorm[:, j], target_denorm[:, j])[0, 1]
    print(f"{key:20s} {rmse:10.4f} {corr:12.4f}")


## ---------------------------------------------------------------------
## Example predictions for a few individual samples
## ---------------------------------------------------------------------
print("\nExample predictions (first 5 validation samples):\n")
for i in range(min(5, pred_denorm.shape[0])):
    print(f"--- sample {i} ---")
    for j, key in enumerate(target_keys[:pred_denorm.shape[1]]):
        print(f"  {key:20s} pred: {pred_denorm[i, j]:10.4f}   "
              f"true: {target_denorm[i, j]:10.4f}")
