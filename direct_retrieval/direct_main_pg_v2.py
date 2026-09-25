"""
Training script for the "direct retrieval" model: predicts 13 microphysical/
turbulence parameters directly from dual-frequency (W-band + Ka-band) Doppler
spectra. This is the commented version of direct_main.py, with an explicit
final checkpoint save added at the end of training (in addition to the
per-epoch saves that were already there).
"""

from torch.utils.data import DataLoader
import dataloader_direct as dataloader
from direct_models import *          # brings in ResBlock, Direct_w_Resblock_seq, etc.
import torch
import numpy as np
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch import nn
from torch.nn import functional as F
import os
import shutil
import json
import matplotlib.pyplot as plt
import time

from torch.utils.tensorboard import SummaryWriter

# Avoids a known PyTorch multiprocessing issue with too many open file
# descriptors when using num_workers > 0 in the DataLoader.
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import argparse


## ---------------------------------------------------------------------
## Parse input and load config
## ---------------------------------------------------------------------
# The script is meant to be run as: python direct_main.py my_config.json
parser = argparse.ArgumentParser(description='train Direct model')
parser.add_argument('config_file', type=str)
args = parser.parse_args()

config = json.load(open(args.config_file))

# Directory where checkpoints, TensorBoard logs, and a copy of the config
# used for this run will be stored.
directory = config['directory']
if not (os.path.exists(directory)):
    os.mkdir(directory)

# Keep a copy of the exact config used for this run, alongside the outputs,
# for reproducibility. Skips silently if the config already lives there.
try:
    shutil.copy2(args.config_file, directory)
except shutil.SameFileError:
    pass


## ---------------------------------------------------------------------
## Set-up TensorBoard writer
## ---------------------------------------------------------------------
# Logs scalars (training/validation loss) that can be viewed with
# `tensorboard --logdir directory`.
writer = SummaryWriter(log_dir=directory)


## ---------------------------------------------------------------------
## Prepare datasets
## ---------------------------------------------------------------------
# Training set: spectra + target parameters, sliced [i_start_ds : i_end_ds]
# from the shared target_array / spectrum_h5 files.
train_dataset = dataloader.direct_dataset(
    config['target_array'], config['spectrum_h5'],
    i_start_ds=config['i_start_ds'], i_end_ds=config['i_end_ds'],
    mean_normalization_json=config['mean_normalization_json'],
    std_normalization_json=config['std_normalization_json'],
    mean_normalization_npy=config['mean_normalization_npy'],
    std_normalization_npy=config['std_normalization_npy'],
    normalize_spectra=config['normalize_spectra'],
    normalize_target=config['normalize_target']
)

# Validation set: same files, different (non-overlapping) index range
# [i_start_val_ds : i_end_val_ds]. Used to check generalization, never
# trained on.
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

# If True, each epoch trains on a rolling subset of ts_size_per_epoch
# samples rather than the full training set (useful for very large datasets
# where a full epoch would be too slow/large).
use_subsampler = config['use_subsampler']
with_wind = config['with_wind']   # (read from config; not otherwise used below)


## ---------------------------------------------------------------------
## Training parameters, read from config
## ---------------------------------------------------------------------
nb_epochs = config['nb_epochs']
batch_size = config['batch_size']
val_batch_size = config['val_batch_size']
lr_initial = config['learning_rate_ini']

model_name = config['model_name']     # e.g. "Direct_w_Resblock_seq"
n_channels = config['n_channels']     # number of internal conv channels

use_scheduler = config['scheduler']   # "step", "plateau", or something else (no scheduler)
if use_scheduler == 'step':
    lr_sched_step = config['lr_sched_step']
    lr_sched_gamma = config['lr_sched_gamma']
elif use_scheduler == 'plateau':
    lr_factor = config['lr_factor']
    lr_patience = config['lr_patience']
    lr_threshold = config['lr_threshold']
    lr_min = config['lr_min']

# Full path where the "latest" checkpoint gets saved/overwritten each epoch.
checkpoint_name = directory + config['checkpoint_name']

ts_size_per_epoch = config['ts_size_per_epoch']
num_workers_loader = config['num_workers_loader']


## ---------------------------------------------------------------------
## Set-up model, optimizer and scheduler
## ---------------------------------------------------------------------
device = config['device']   # "cpu" or "cuda"

# locals()[model_name] dynamically looks up the class by name (must be one
# of the classes imported from direct_models via `from direct_models import *`)
model = locals()[model_name](n_channels).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr_initial)

if use_scheduler == 'step':
    # Multiplies LR by lr_sched_gamma every lr_sched_step epochs.
    scheduler = StepLR(optimizer, step_size=lr_sched_step, gamma=lr_sched_gamma)
elif use_scheduler == 'plateau':
    # Reduces LR by lr_factor if the monitored loss doesn't improve for
    # lr_patience epochs. NOTE: below, this is stepped using the TRAINING
    # loss (epochloss), not the validation loss — worth double-checking
    # that's actually intended, since ReduceLROnPlateau is usually driven
    # by validation loss.
    scheduler = ReduceLROnPlateau(optimizer, factor=lr_factor, patience=lr_patience,
                                   threshold=lr_threshold, min_lr=lr_min)

# Epoch after which the scheduler stops reducing the learning rate further.
stop_decreasing_after = config['stop_decreasing_after']

# Optional weight initialization scheme.
init = config['init']
if init == 'xavier_normal':
    def weights_init(m):
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    model.apply(weights_init)


## ---------------------------------------------------------------------
## Load the checkpoint if it exists (resume training)
## ---------------------------------------------------------------------
bc = 0                       # running count of total batches processed (across epochs)
nb_epochs_finished = 0
N = len(train_dataset)
epoch_loss_previous = 0

# Plain Python lists tracking loss per epoch, so we can plot the loss
# curve directly in this script without needing to parse TensorBoard's
# binary log files. If resuming from a checkpoint, load the existing
# history so the curve stays continuous across runs.
loss_history_path = checkpoint_name[:-4] + '_loss_history.npz'
try:
    _hist = np.load(loss_history_path)
    train_loss_history = list(_hist['train_loss'])
    val_loss_history = list(_hist['val_loss'])
    print(f'Loaded existing loss history ({len(train_loss_history)} epochs).')
except FileNotFoundError:
    train_loss_history = []
    val_loss_history = []

try:
    checkpoint = torch.load(checkpoint_name)
    nb_epochs_finished = checkpoint['nb_epochs_finished']
    bc = checkpoint['nb_batches_finished']
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    if (use_scheduler == 'step') & (nb_epochs_finished <= stop_decreasing_after):
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    print(f'Checkpoint loaded with {nb_epochs_finished} epochs finished.')
except FileNotFoundError:
    # No checkpoint yet at this path -> train from scratch, starting at epoch 0.
    print('Starting from scratch.')
except Exception as e:
    # Any other error while loading (corrupted file, shape mismatch, etc.)
    # is treated as fatal, since silently continuing could train on top of
    # a broken/mismatched state.
    print('Error when loading the checkpoint: ', e)
    exit(1)


## ---------------------------------------------------------------------
## Perform the training
## ---------------------------------------------------------------------
# Validation dataloader is fixed for the whole run (val set doesn't change
# between epochs).
val_dataloader = DataLoader(val_dataset, num_workers=num_workers_loader,
                             batch_size=val_batch_size, pin_memory=True)

# If not using the subsampler, the training dataloader is also fixed for
# the whole run (iterates the full training set every epoch).
if not (use_subsampler):
    train_dataloader = DataLoader(train_dataset, num_workers=num_workers_loader,
                                   batch_size=batch_size, pin_memory=True)

for epoch in range(nb_epochs_finished, nb_epochs):
    epochloss = 0

    if use_subsampler:
        # Carve out a rolling window of ts_size_per_epoch indices from the
        # training set for this epoch, wrapping around with modulo once
        # the window passes the end of the dataset. This lets you train on
        # a large dataset without a full pass over all of it every epoch.
        inds = torch.arange(
            (ts_size_per_epoch * epoch) % N,
            min(N, (ts_size_per_epoch * epoch) % N + ts_size_per_epoch)
        )
        train_dataloader = DataLoader(
            train_dataset, num_workers=num_workers_loader, batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(inds), pin_memory=True
        )
        print(len(train_dataloader))

    ## --- one training epoch: loop over batches ---
    for (input_data, target) in iter(train_dataloader):
        input_data = input_data.to(device)
        target = target.to(device)

        # Forward pass: predict the 13 target parameters from the spectra.
        output = model(input_data)

        # Loss: 0.5 * sum-of-squared-errors, averaged over the batch.
        # (This is a scaled MSE — the 0.5 factor is a common convention
        # that cancels out nicely when differentiating x^2.)
        loss = 0.5 * ((output[:, 0, :] - target).pow(2).sum()) / output.size(0)

        epochloss += loss * output.size(0)

        # Standard PyTorch training step: clear old gradients, backprop,
        # update weights. THIS is where the weights actually get updated —
        # once per batch, not once per epoch.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if bc % 5 == 0:
            writer.add_scalar('batch_loss', loss, bc / 5)
            print(epoch, bc, loss)
        bc += 1

        # (disabled) periodically log a plot comparing predicted vs. true
        # spectrum/target for visual inspection during training
        # fig = plt.figure(figsize=(10,3))
        # plt.plot(output[0][:].cpu().detach().numpy())
        # plt.plot(target[0][:].cpu().detach().numpy())
        # writer.add_figure('spectrum_train',fig,epoch)
        # plt.close()

    ## --- validation pass, once per epoch, no gradient updates ---
    with torch.no_grad():
        val_epochloss = 0
        for (input_data, target) in iter(val_dataloader):
            input_data = input_data.to(device)
            target = target.to(device)
            output = model(input_data)
            val_loss = 0.5 * ((output[:, 0, :] - target).pow(2).sum()) / output.size(0)
            val_epochloss += val_loss * output.size(0)
        val_epochloss = val_epochloss / len(val_dataset)

    writer.add_scalar('training_loss', epochloss / ts_size_per_epoch, epoch)
    writer.add_scalar('validation_loss', val_epochloss, epoch)

    # Track the same two numbers in plain Python lists, for plotting below.
    train_loss_history.append(float(epochloss / ts_size_per_epoch))
    val_loss_history.append(float(val_epochloss))

    # (disabled) histogram logging of weights/gradients per layer
    # def writer_histogram(m):
    #     if ((isinstance(m, nn.Conv1d)) | (isinstance(m, nn.ConvTranspose1d)) | (isinstance(m,nn.Linear))):
    #         writer.add_histogram(str(m)+'.weight', m.weight, epoch)
    #         writer.add_histogram(str(m)+'.weight.grad', m.weight.grad, epoch)
    # model.apply(writer_histogram)

    # (disabled) periodically log a plot comparing predicted vs. true
    # spectrum/target on validation data
    # fig = plt.figure(figsize=(10,3))
    # plt.plot(output[0][:].cpu().detach().numpy())
    # plt.plot(target[0][:].cpu().detach().numpy())
    # writer.add_figure('spectrum_val',fig,epoch)
    # plt.close()

    epoch_loss_previous = epochloss / ts_size_per_epoch

    ## --- checkpoint: save model/optimizer/scheduler state after this epoch ---
    checkpoint = {
        'nb_epochs_finished': epoch + 1,
        'nb_batches_finished': bc + 1,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }

    if (use_scheduler == 'step') & (epoch <= stop_decreasing_after):
        scheduler.step()
        checkpoint['scheduler_state'] = scheduler.state_dict()
    elif (use_scheduler == 'plateau') & (epoch <= stop_decreasing_after):
        scheduler.step(epochloss)
        checkpoint['scheduler_state'] = scheduler.state_dict()

    # Overwrite the "latest" checkpoint every epoch, so training can always
    # resume from the most recent state if interrupted.
    torch.save(checkpoint, checkpoint_name)

    # Save the loss history every epoch too, so it survives an interrupted
    # run just like the checkpoint does.
    np.savez(loss_history_path,
             train_loss=np.array(train_loss_history),
             val_loss=np.array(val_loss_history))

    # Re-plot and save the loss curve as a PNG every epoch, so you can open
    # it at any point (even mid-run, or after an interruption) without a
    # separate script or TensorBoard.
    loss_plot_path = checkpoint_name[:-4] + '_loss.png'
    fig_loss, ax_loss = plt.subplots(figsize=(8, 5))
    epochs_so_far = np.arange(len(train_loss_history))
    ax_loss.plot(epochs_so_far, train_loss_history, label='training loss')
    ax_loss.plot(epochs_so_far, val_loss_history, label='validation loss')
    best_epoch = int(np.argmin(val_loss_history))
    ax_loss.axvline(best_epoch, color='gray', linestyle='--', alpha=0.5,
                     label=f'best val epoch ({best_epoch})')
    ax_loss.set_xlabel('epoch')
    ax_loss.set_ylabel('loss')
    ax_loss.set_yscale('log')
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)
    ax_loss.set_title('Training / validation loss evolution')
    fig_loss.tight_layout()
    fig_loss.savefig(loss_plot_path, dpi=150)
    plt.close(fig_loss)

    # Additionally keep a permanent, non-overwritten snapshot every 20
    # epochs, in case you want to go back to an earlier point later
    # (e.g. if a later epoch overfits).
    if (epoch + 1) % 2 == 0:
        torch.save(checkpoint, checkpoint_name[:-4] + '_' + str(epoch) + '.pth')
## ---------------------------------------------------------------------
## Explicit final save (new addition)
## ---------------------------------------------------------------------
# The loop above already saves a checkpoint every epoch, so by the time
# training finishes normally, checkpoint_name already holds the final
# state. This block makes that explicit and also writes a clearly-named
# "final" copy, so you don't have to guess which epoch-numbered file (if
# any) corresponds to the very end of training.
final_checkpoint = {
    'nb_epochs_finished': nb_epochs,
    'nb_batches_finished': bc,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
}
if use_scheduler in ('step', 'plateau'):
    final_checkpoint['scheduler_state'] = scheduler.state_dict()

final_checkpoint_path = checkpoint_name[:-4] + '_final.pth'
torch.save(final_checkpoint, final_checkpoint_path)
print(f'Training complete. Final model saved to: {final_checkpoint_path}')
print(f'(Latest per-epoch checkpoint also available at: {checkpoint_name})')
print(f'(Loss curve saved to: {checkpoint_name[:-4]}_loss.png, updated every epoch)')
