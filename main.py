import os
import torch
import yaml
import logging 
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from data.Custom_image_dataset import dataset
from train import train_step, validation_step
import datetime
import wandb
import h5py
import numpy as np
import argparse
from arch.hamba import *
import wandb
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm import utils
from timm.scheduler import *

torch.backends.cudnn.benchmark = True
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def get_norm_layer(layer_name):
    """Maps string names to PyTorch normalization classes."""
    if layer_name == 'LayerNorm':
        return nn.LayerNorm
    elif layer_name == 'BatchNorm':
        return nn.BatchNorm2d
    elif layer_name == 'Identity':
        return nn.Identity
    else:
        raise NotImplementedError(f"Normalization layer {layer_name} is not found")
def get_activation(act):
    if act=='GELU':
        return nn.GELU
    else:
        raise NotImplementedError(f"Normalization layer {layer_name} is not found")
def load_config_and_parse_args():
    # --- 1. Initial Parser to get the config file path ---
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                                help='YAML config file specifying default arguments')
    
    # Parse just the config file path, leaving other arguments for the main parser
    config_args, remaining_argv = config_parser.parse_known_args()

    defaults = {}
    if config_args.config and os.path.exists(config_args.config):
        print(f"Configuration loaded from YAML: {config_args.config}")
        with open(config_args.config, 'r') as f:
            # Load all YAML content into a single flat dictionary
            defaults = yaml.safe_load(f)
    else:
        # If no config file is provided or found, this message is useful for debugging
        print("No valid YAML configuration file loaded. Using hardcoded defaults.")


    # --- 2. Main Parser Definition ---
    # The 'parents=[config_parser]' ensures the -c/--config argument is still available for help/usage info
    parser = argparse.ArgumentParser(description='HAMBA Super-Resolution Training', parents=[config_parser])
    
    # Helper to safely get values from the flat defaults dictionary
    def get_default(key, default_val=None):
        return defaults.get(key, default_val)

    # =========================================================================
    # A. GENERAL & ENVIRONMENT
    # =========================================================================
    parser.add_argument('--name', default=get_default('name', 'HAMBA_X4'), type=str, help='Experiment name.')
    parser.add_argument('--num_gpu', default=get_default('num_gpu', 'auto'), type=str, help='Number of GPUs or "auto".')
    parser.add_argument('--manual_seed', default=get_default('manual_seed', 0), type=int, help='Random seed.')
    parser.add_argument('--resume_training', action='store_true', default=get_default('resume_training', False), help='Resume training from checkpoint.')
    parser.add_argument('--device', default=get_default('device', 'cuda'), type=str, help='Device to run on (e.g., cuda, cpu).')


    # =========================================================================
    # B. DATASET & DATALOADERS
    # =========================================================================
    parser.add_argument('--train_file_pth', default=get_default('train_file_pth'), type=str, help='Path to training data H5 file.')
    parser.add_argument('--val_file_pth', default=get_default('val_file_pth'), type=str, help='Path to validation data H5 file.')
    parser.add_argument('--test_file_pth', default=get_default('test_file_pth'), type=str, help='Path to test data H5 file.')
    parser.add_argument('--train_batch_size', default=get_default('train_batch_size', 4), type=int, help='Training batch size.')
    parser.add_argument('--train_shuffle', action=argparse.BooleanOptionalAction, default=get_default('train_shuffle', True), help='Shuffle training data.')
    parser.add_argument('--val_batch_size', default=get_default('val_batch_size', 1), type=int, help='Validation batch size.')
    parser.add_argument('--val_shuffle', action=argparse.BooleanOptionalAction, default=get_default('val_shuffle', False), help='Shuffle validation data.')
    parser.add_argument("--train_indices", default= get_default('train_indices'), type= str, help = "Training indices")
    parser.add_argument("--val_indices", default= get_default('val_indices'), type= str, help = "Validation indices")
    # =========================================================================
    # C. NETWORK (HAMBA) PARAMETERS
    # =========================================================================
    parser.add_argument('--model_type', type=str, default='HAMBA')
    parser.add_argument('--in_chans', type=int, default=3)
    parser.add_argument('--input_resolution', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=96)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_block', type=int, default=6)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--window_size', type=int, default=8)
    parser.add_argument('--overlap_ratio', type=float, default=0.5)
    parser.add_argument('--upscale', type=int, default=4)
    parser.add_argument('--upsampler', type=str, default='pixelshuffle')
    parser.add_argument('--img_range', type=float, default=1.)
    parser.add_argument('--shift_size', type=int, default=0)
    parser.add_argument('--compress_ratio', type=int, default=3)
    parser.add_argument('--squeeze_factor', type=int, default=30)
    parser.add_argument('--drop_rate', type=float, default=0.)
    parser.add_argument('--conv_scale', type=float, default=0.01)
    parser.add_argument('--mlp_ratio', type=float, default=4.)
    parser.add_argument('--qkv_bias', type=str2bool, default=True)
    parser.add_argument('--qk_scale', type=float, default=None)
    parser.add_argument('--patch_norm', type=str2bool, default=True)
    parser.add_argument('--ape', type=str2bool, default=False)
    parser.add_argument('--drop', type=float, default=0.)
    parser.add_argument('--attn_drop', type=float, default=0.)
    parser.add_argument('--drop_path', type=float, default=0.)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--d_conv', type=int, default=4)
    parser.add_argument('--expand', type=int, default=2)
    parser.add_argument('--dt_rank', type=str, default="auto")
    parser.add_argument('--dt_min', type=float, default=0.001)
    parser.add_argument('--dt_max', type=float, default=0.1)
    parser.add_argument('--dt_init', type=str, default="random")
    parser.add_argument('--dt_scale', type=float, default=1.0)
    parser.add_argument('--dt_init_floor', type=float, default=1e-4)
    parser.add_argument('--conv_bias', type=str2bool, default=True)
    parser.add_argument('--bias', type=str2bool, default=False)
    parser.add_argument('--use_fast_path', type=str2bool, default=True)
    parser.add_argument('--layer_idx', default=None)
    parser.add_argument('--dtype', default=None)
    parser.add_argument('--norm_layer', default='LayerNorm', type= get_norm_layer)
    parser.add_argument('--act', default='GELU', type = get_activation)
    # =========================================================================
    # D. TRAINING & OPTIMIZER
    # =========================================================================
    parser.add_argument('--epoch', default=get_default('epoch', 200), type=int, help='Total number of training epochs.')
    parser.add_argument('--lr', default=get_default('lr', 1e-5), type=float, help='Generator learning rate.')
    parser.add_argument('--optim_type', default=get_default('type', 'Adam'), type=str, help='Optimizer type.')
    parser.add_argument('--weight_decay', default=get_default('weight_decay', 0), type=float, help='Optimizer weight decay.')
    parser.add_argument('--betas', default=get_default('betas', [0.9, 0.99]), nargs='+', type=float, help='Adam beta values.')
    parser.add_argument('--scheduler_type', default=get_default('type', 'MultiStepLR'), type=str, help='LR scheduler type.')
    parser.add_argument('--milestones', default=get_default('milestones', [25, 50, 75, 100, 125, 150, 175, 180]), nargs='+', type=int, help='Scheduler milestones.')
    parser.add_argument('--gamma', default=get_default('gamma', 0.5), type=float, help='Scheduler gamma value.')

    # =========================================================================
    # E. LOSS & PATHS
    # =========================================================================
    parser.add_argument('--loss_type', default=get_default('type', 'L1Loss'), type=str, help='Loss function type.')
    parser.add_argument('--loss_weight', default=get_default('loss_weight', 1.0), type=float, help='Loss weight.')
    parser.add_argument('--results_path', default=get_default('results'), type=str, help='Directory for saving results.')
    parser.add_argument('--checkpoint_path', default=get_default('checkpoint'), type=str, help='Directory for saving checkpoints.')

    # --- 3. Final Parsing ---
    args = parser.parse_args(remaining_argv)
    
    return args

def main():
    utils.setup_default_logging()
    local_rank = int(os.environ.get("SLURM_LOCALID")) 
    rank = int(os.environ.get("SLURM_PROCID"))
    num_workers= 8
    world_size = int(os.environ.get("SLURM_NTASKS"))
    current_device = local_rank
    torch.cuda.set_device(current_device)
    init_process_group(backend='nccl', world_size=world_size, rank=rank, timeout= datetime.timedelta(seconds=7200))
    args = load_config_and_parse_args()
    epochs= args.epoch
    device = torch.device(f"cuda:{local_rank}")
    run = wandb.init("Hamba Model",config = args)
    print(type(args.embed_dim))
    model = HAMBA(
        in_chans= args.in_chans, 
        input_resolution=args.input_resolution,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_block=args.num_block,
        patch_size=args.patch_size,
        img_size=args.img_size, 
        window_size=args.window_size,
        overlap_ratio= args.overlap_ratio,
        norm_layer=args.norm_layer,
        act_layer= args.act, 
        upscale=args.upscale, 
        upsampler=args.upsampler, 
        img_range=args.img_range,
        shift_size=args.shift_size,
        compress_ratio=args.compress_ratio,
        squeeze_factor=args.squeeze_factor,
        drop_rate=args.drop_rate, 
        drop_path=args.drop_path, 
        d_state=args.d_state, 
        d_conv=args.d_conv, 
        expand=args.expand,
        dt_rank=args.dt_rank, 
        dt_min=args.dt_min, 
        dt_max=args.dt_max, 
        dt_init=args.dt_init, 
        dt_scale=args.dt_scale, 
        dt_init_floor=args.dt_init_floor, 
        conv_bias=args.conv_bias, 
        bias= args.bias, 
        use_fast_path=args.use_fast_path, 
        layer_idx=args.layer_idx, 
        device= args.device, 
        dtype= args.dtype
    )#.to(device)

    #Build Optimizers and loss functions and schedulers
    loss_fn = nn.L1Loss()#.to(device)
    optimizer= torch.optim.Adam(params= model.parameters(), lr = args.lr, betas= (args.betas))#.to(device)
    lr_scheduler= torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones)

    # Build Dataset and dataloaders
    train_data = dataset(args.train_file_pth, args.train_indices)
    train_sampler = DistributedSampler(train_data, shuffle=args.train_shuffle)
    train_loader= DataLoader(
        train_data, 
        batch_size= args.train_batch_size, 
        sampler= train_sampler,
        num_workers= 8
    )
    val_data= dataset(args.val_file_pth, args.val_indices)
    val_sampler= DistributedSampler(val_data, shuffle=args.val_shuffle)
    val_loader= DataLoader(
        val_data, 
        batch_size= args.val_batch_size, 
        sampler= val_sampler,
        num_workers= 8
    )
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        train_loss= train_step(model, loss_fn, optimizer, train_loader, device, epoch, rank ,  world_size)
        if local_rank == 0 :
            run.log({"Train loss": train_loss})
        if epoch % 10 == 0:
            if val_loader is not None:
                # ALL processes run validation
                val_loss, PSNR, SSIM = validation_step(model, loss_fn, val_loader, device, epoch, rank,world_size)
                if rank == 0:
                    run.log({"Validation loss": val_loss, 
                             "PSNR": PSNR, 
                             "SSIM": SSIM})
                    torch.save(
                        model.module.state_dict(),
                        # config['checkpoint'] + f"/Imagenet_{epoch}.pth"
                    )

            elif rank == 0: 
                torch.save(
                    model.module.state_dict(),
                    # config['checkpoint'] + f"/Imagenet_{epoch}.pth"
                )
        lr_scheduler.step()
    if rank == 0:
        wandb.finish()
    destroy_process_group()
if __name__== "__main__":
    main()