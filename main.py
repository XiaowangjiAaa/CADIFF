#%%
import sys
sys.path.append("..")
#%%
import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ISIC_aug_dataset
# from torchmetrics import Dice
from tqdm.notebook import tqdm
import pickle
import torch.nn.functional as f
from monai.losses import DiceLoss
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
#%%
from improved_diffusion.ss_unet import UNetModel_WithSSF
from improved_diffusion.script_util import create_gaussian_diffusion
from improved_diffusion.resample import UniformSampler
from gan import NLayerDiscriminator
#%%
image_size = 256
batch_size = 8
epochs = 100
# DEVICE will be updated after Accelerator initialization
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
path = "E:\dataset\ISIC_augmentation"
save_path = "./final_result"
dis_save_path = "./final_result/dis"
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(dis_save_path):
    os.makedirs(dis_save_path)
#%%
# unet hyper parameterts
model_channnels = 128
in_channels = 4 # 1+3
out_channels = 1
num_res_blocks = 1
attn_resolutions = [] # if use, default is [16]
dropout = 0.0
channel_mult = (1, 1, 2, 2, 4, 4) if image_size == 256 else None
dims = 2
num_classes = None
num_heads = 4 # not used in model
num_heads_upsample = -1 # not used in model
use_checkpoint = False
use_scale_shift_norm = False

num_train_D = 2 # set this
num_train_D += 1
time_adv_thr = 100
loss_adv_weight = 0.5

dicefunc = DiceLoss()
#%%
# diffusion hyper parameters
steps = 1000
learn_sigma = False
predict_xstart = False
#%%
def ls_gan_loss(dis, pred, target, timesteps, mode:str):
    assert mode in ['train_G','train_D']
    B = pred.shape[0]
    weights = torch.where(timesteps < time_adv_thr, 1., 0.)
    sum = weights.sum()
    if sum > 0:
        while len(weights.shape) < len(pred.shape):
            weights.unsqueeze_(-1)
        idx = torch.where(weights > 0)
        idx=idx[0]
        pred = torch.index_select(pred, 0, idx)
        target = torch.index_select(target, 0, idx)
        D = dis
        if mode == 'train_G':
            output_pred = D(pred)
            loss_adv = f.mse_loss(output_pred, torch.ones_like((output_pred)))
        if mode == 'train_D':
            output_pred = D(pred.detach())
            output_target = D(target.float())
            loss_adv = f.mse_loss(output_target, torch.ones_like((output_target))) \
                + f.mse_loss(output_pred, torch.zeros_like(output_pred))
        return loss_adv * 0.5
    else:
        return 0
#%%
Diff_UNet = UNetModel_WithSSF(
    model_channels=model_channnels,
    in_channels=in_channels,
    out_channels=out_channels,
    channel_mult=channel_mult,
    num_res_blocks=num_res_blocks,
    attention_resolutions=attn_resolutions,
    dropout=dropout,
    dims=dims,
    num_classes=num_classes,
    num_heads=num_heads,
    num_heads_upsample=num_heads_upsample,
    use_checkpoint=use_checkpoint,
    use_scale_shift_norm=use_scale_shift_norm,
)
Diff_UNet.load_resunet(if_pre=False, in_channels=3)
discriminator = NLayerDiscriminator(input_nc=1)
#%%
resume = False
if resume:
    state_dict = torch.load(os.path.join(save_path, "diff_unet_v1_withgan_withss.pt"))
    Diff_UNet.load_state_dict(state_dict)
    dis_state_dict = torch.load(os.path.join(dis_save_path, "dis.pt"))
    discriminator.load_state_dict(dis_state_dict)
#%%
diffusion = create_gaussian_diffusion(steps=1000, learn_sigma=False, predict_xstart=False)
sampler = UniformSampler(diffusion)

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs])
accelerator.init_trackers("cadiiff")
DEVICE = accelerator.device
#%%
isic_train = ISIC_aug_dataset(path = path, type = 'Train', image_size=256)
# isic_test = ISIC_aug_dataset(path = path, type = 'test', image_size=256)

train_loader = DataLoader(isic_train, batch_size=8, shuffle=True)
# test_loader = DataLoader(isic_test, batch_size=1, shuffle=False)
#%%
opt_seg = torch.optim.AdamW(Diff_UNet.parameters(), lr=1e-4, betas=(0.5, 0.999))
opt_d = torch.optim.AdamW(discriminator.parameters(), lr=1e-4, betas=(0., 0.999))
lr_schedular_seg = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_seg, T_0=7, T_mult=2)
# lr_schedular_d=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_d, T_0=7, T_mult=2)
history = {'loss_diff': [], 'loss_D': [], 'loss_G': [], 'loss_dice': [], 'loss_mse': []}

lossfunc1 = nn.MSELoss()

Diff_UNet, discriminator, opt_seg, opt_d, train_loader = accelerator.prepare(
    Diff_UNet, discriminator, opt_seg, opt_d, train_loader
)

# Freeze parameters not present in optimizer to avoid errors with Accelerate
opt_params = {p for group in opt_seg.param_groups for p in group["params"]}
unused = [n for n, p in Diff_UNet.named_parameters() if p.requires_grad and p not in opt_params]
if unused:
    for name, p in Diff_UNet.named_parameters():
        if name in unused:
            p.requires_grad_(False)
    accelerator.print(f"Frozen unused parameters: {unused}")
#%%
# initial for train
trainstep=(len(train_loader.dataset)//batch_size)+1 
outtertqdm=tqdm(range(epochs))
best_loss=100
mode_num = 0
mode = ['train_G', 'train_D']

for epoch in outtertqdm:
    # initial for each epoch
    innertqdm=tqdm(range(trainstep),leave=False)
    dataiter=iter(train_loader)
    Diff_UNet.train()
    totalLoss_diff = 0
    totalLoss_gan_D = 0
    totalLoss_gan_G = 0
    totalLoss_dice = 0
    totalLoss_mse = 0
    step = 0
    step_D = 0
    step_G = 0
    
    
    for _ in innertqdm:
        step += 1
        mode_num += 1
        # initial for each step
        (img,real_mask)=next(dataiter)
        (img,real_mask)=(img.to(DEVICE),real_mask.to(DEVICE))
        t, weights = sampler.sample(img.shape[0], DEVICE)
        opt_seg.zero_grad()
        opt_d.zero_grad()

        #####Loss#####
        noise = torch.randn_like(real_mask)
        x_t = diffusion.q_sample(real_mask, t, noise=noise)
        i_t = torch.cat([x_t, img], dim=1)
        pred = Diff_UNet(i_t, diffusion._scale_timesteps(t), img)
        pred_xstart = diffusion._predict_xstart_from_eps(x_t, t, pred)
        loss = lossfunc1(pred, noise)
        predxstart_clip = torch.clamp(pred_xstart, 0, 1)
        loss_dice = dicefunc(predxstart_clip, real_mask)
        loss_mse = f.mse_loss(pred_xstart, real_mask)

        totalLoss_dice += loss_dice
        totalLoss_mse += loss_mse
        totalLoss_diff += loss
        ##### GAN Loss #####
        gan_loss = ls_gan_loss(discriminator, pred_xstart, real_mask, t, mode[1 if mode_num % num_train_D else 0])

        if gan_loss > 0:
            if mode_num % num_train_D:
                totalLoss_gan_D += gan_loss
                accelerator.log({'loss_D': float(gan_loss.cpu().detach().numpy())}, step=epoch * trainstep + step)
                step_D += 1
            else:
                totalLoss_gan_G += gan_loss
                accelerator.log({'loss_G': float(gan_loss.cpu().detach().numpy())}, step=epoch * trainstep + step)
                step_G += 1

        # update
        total_loss = loss + loss_dice + gan_loss * loss_adv_weight + loss_mse
        accelerator.backward(total_loss)
        
        opt_seg.step()
        if mode[1 if mode_num % num_train_D else 0] == 'train_D':
            opt_d.step()
            
        innertqdm.set_postfix({'step': step + 1, 'loss': loss.cpu().detach().numpy().item()})
        accelerator.log({
            'loss_diff': float(loss.cpu().detach().numpy()),
            'loss_dice': float(loss_dice.cpu().detach().numpy()),
            'loss_mse': float(loss_mse.cpu().detach().numpy())
        }, step=epoch * trainstep + step)


    avgLoss_diff=totalLoss_diff.cpu().detach().numpy()/step
    avgLoss_dice=totalLoss_dice.cpu().detach().numpy()/step
    avgLoss_mse=totalLoss_mse.cpu().detach().numpy()/step
    avgLoss_G=totalLoss_gan_G.cpu().detach().numpy()/step_G
    avgLoss_D=totalLoss_gan_D.cpu().detach().numpy()/step_D

    history['loss_diff'].append(avgLoss_diff)
    history['loss_G'].append(avgLoss_G)
    history['loss_D'].append(avgLoss_D)
    history['loss_dice'].append(avgLoss_dice)
    history['loss_mse'].append(avgLoss_mse)

    outtertqdm.set_postfix({'Epoch': epoch+1, 'Loss':avgLoss_diff.item()})

    if best_loss>avgLoss_diff:
        best_loss=avgLoss_diff
        torch.save(Diff_UNet.state_dict(), os.path.join(save_path, 'diff_unet_v1_withgan_withss_best.pt'))
    torch.save(Diff_UNet.state_dict(), os.path.join(save_path, 'diff_unet_v1_withgan_withss.pt'))
    torch.save(discriminator.state_dict(), os.path.join(save_path, 'dis','dis.pt'))

    lr_schedular_seg.step()
    # lr_schedular_d.step()

with open(os.path.join(save_path, 'history_resunet.pkl'),'wb') as f:
    pickle.dump(history,f)

accelerator.end_training()

