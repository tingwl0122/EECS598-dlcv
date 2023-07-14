import math
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn
import skimage
from skimage import color
from skimage import metrics
import numpy as np
from PIL import Image
import torchvision
import os
import time



# convert rgb to ycbcr

def convert_rgb2ycbcr(img):
    # input image size: (B, C, H, W)
    # permute to make channel the last dimension
    if img.dim() == 4:
      img_np = skimage.color.rgb2ycbcr(img.permute(0,2,3,1).contiguous())
      result = torch.from_numpy(img_np).permute(0,3,1,2).contiguous()
    
    elif img.dim() == 3:
      img_np = skimage.color.rgb2ycbcr(img.permute(1,2,0).contiguous())
      result = torch.from_numpy(img_np).permute(2,0,1).contiguous()
    return result
# convert rbg to y
def convert_rgb2y(img):
    # input image size: (B, C, H, W)
    # permute to make channel the last dimension
    if img.dim() == 4:
      img_np = skimage.color.rgb2ycbcr(img.permute(0,2,3,1).contiguous())
      # output shape
      result = torch.from_numpy(img_np).permute(0,3,1,2).contiguous()[:,0:1,:,:]
    
    elif img.dim() == 3:
      img_np = skimage.color.rgb2ycbcr(img.permute(1,2,0).contiguous())
      result = torch.from_numpy(img_np).permute(2,0,1).contiguous()[0:1,:,:]
    return result


def data_augmentation(img):
    #img = torch.unsqueeze(img, 0)
    degrees = [0, 90, 180, 270]
    resizes = [0.6, 0.7, 0.8, 0.9, 1]
    batch = len(degrees) * len(resizes)
    #data_ag = torch.zeros(batch, 3, 112, 112, dtype=torch.uint8)
    data_ag = []
    #i = 0
    for deg in degrees:
        for rsz in resizes:
            img_ag = torchvision.transforms.functional.rotate(img, angle=deg) # 0, 180, 90, 270
            w, h = img.shape[1], img.shape[2] 
            
            img_scale = torchvision.transforms.Resize((int(w*rsz), int(h*rsz)), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
            
            img_ag = img_scale(img_ag)
            
            data_ag.append(img_ag)
    return data_ag

def crop_patch(hr_images, hr_patches, lr_patches, scale_factor, patch_size, stride, w, h):

    for hr_img in hr_images:
        #w, h = img.shape[1], img.shape[2] 
        hr_width = (w // scale_factor) * scale_factor
        hr_height = (h // scale_factor) * scale_factor
        hr_transform = torchvision.transforms.Resize((hr_width, hr_height), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        hr = hr_transform(hr_img)
        
        lr_transform = torchvision.transforms.Resize((hr_width // scale_factor, hr_height // scale_factor), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        lr = lr_transform(hr)
        
        lr = convert_rgb2y(lr)
        hr = convert_rgb2y(hr)
        


    

        for i in range(0, lr.shape[1] - patch_size + 1, stride):
            for j in range(0, lr.shape[2] - patch_size + 1, stride):
                
                lr_patches.append(lr[:, i : i + patch_size, j : j + patch_size].unsqueeze(0))
                hr_patches.append(hr[:,i * scale_factor : i * scale_factor + patch_size * scale_factor, j * scale_factor : \
                                    j * scale_factor + patch_size * scale_factor].unsqueeze(0))



def get_validate_img(path, scaling_factor):
    if scaling_factor == 2:
        filenames = [f'img_00{i}_SRF_2_' for i in range(1, 10)] + [f'img_0{i}_SRF_2_' for i in range(10, 100)] + [f'img_100_SRF_2_']
    elif scaling_factor == 3:
        filenames = [f'img_00{i}_SRF_3_' for i in range(1, 10)] + [f'img_0{i}_SRF_3_' for i in range(10, 100)] + [f'img_100_SRF_3_']
    
  
    batch_size = len(filenames)
    eval_in = []
    eval_out = []
    
    for i, filename in enumerate(filenames):
        file_in = filename + 'LR.png'
        file_out = filename + 'HR.png'
        # file in 
        img = Image.open(os.path.join(path, file_in))
        img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
        eval_in.append(img)
        # file out
        img = Image.open(os.path.join(path, file_out))
        #img = img.crop((0, 0, out_dim, out_dim))
        img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
        eval_out.append(img)
        # print(img.shape)
        if (i+1) % 10 == 0:
          print("# processed images: ", i+1)
    return eval_in, eval_out


# Our core FSRCNN model

class FSRCNN(nn.Module):
    """
    the full implementation of FSRCNN model.
        Input:
        - scale_factor: the factor that scales the model 
        - num_channels: input & output channel
        - m: number of conv layers in the mapping stage
        - d: number of channels before shrinking
        - s: number of chnnels during mapping stage

        Create following structures:
        - self.first_part: feature extraction
        - self.mid_part: shrinking, mapping, and expanding part
        - self.last_part: devonolution layer with size (d, num_channels, )
    """
    def __init__(self, scale_factor, num_channels=1, d=56, s=16, m=4, dtype=None, device=torch.device("cuda")):
        super(FSRCNN, self).__init__()

        # feature extraction
        self.first_part = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=d, kernel_size=5, padding=5//2, dtype=dtype, device=device),
            nn.PReLU(d)
        )
        # shrinking
        self.mid_part = [nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, dtype=dtype, device=device), nn.PReLU(s)]
        
        # mapping
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, padding=3//2, dtype=dtype, device=device), nn.PReLU(s)])

        # expanding
        self.mid_part.extend([nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1, dtype=dtype, device=device), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        
        # deconvolution
        self.last_part = nn.ConvTranspose2d(in_channels=d, out_channels=num_channels, kernel_size=9, stride=scale_factor, padding=9//2,
                                            output_padding=scale_factor-1, dtype=dtype, device=device)

        self._initialize_weights()
        self.dtype=dtype
        self.device=device

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)

        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)

        nn.init.normal_(self.last_part.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias)

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x

def train_super_resolution(
    model,
    training_data,
    gt_data,
    val_data,
    val_gt_data,
    scale_factor,
    num_epochs,
    batch_size,
    learning_rate,
    lr_decay=0.99,
    from_scratch=False,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
):
    """
    Run optimization to train the model.
    """

    # training_data = training_data.to(device)
    # gt_data = gt_data.to(device)
    model = model.to(device)
    model.train()

    # optimizer = torch.optim.AdamW(
    #     filter(lambda p: p.requires_grad, model.parameters()), learning_rate
    # )

    if from_scratch:
        optimizer = torch.optim.AdamW([
            {'params': model.first_part.parameters()},
            {'params': model.mid_part.parameters()},
            {'params': model.last_part.parameters(), 'lr': learning_rate * 0.1}
        ], lr=learning_rate)
    else:
        for param in model.first_part.parameters():
            param.requires_grad = False
        for param in model.mid_part.parameters():
            param.requires_grad = False
        optimizer = torch.optim.AdamW([
            {'params': model.last_part.parameters(), 'lr': learning_rate * 0.1}
        ], lr=learning_rate)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: lr_decay ** epoch
    )

    # sample minibatch data
    iter_per_epoch = math.ceil(training_data.shape[0] // batch_size)
    loss_history = []
    loss_cls = torch.nn.MSELoss()

    for i in range(num_epochs):
        start_t = time.time()
        for j in range(iter_per_epoch):
            # get batches of training pairs
            images = training_data[j * batch_size : (j + 1) * batch_size]
            gt_images = gt_data[j * batch_size : (j + 1) * batch_size]
            images = images.to(device)
            gt_images = gt_images.to(device)
            # get prediction by passing our images through the CNN model
            pred_images = model(images)
            # compute MSE loss
            loss = loss_cls(pred_images, gt_images)
            optimizer.zero_grad()
            loss.backward()
            loss_history.append(loss.item())
            optimizer.step()

            
        # evaluate on validation set
        model.eval()

        if val_data:
          with torch.no_grad():
          
            num_val = len(val_data)
            loss_val = 0.0
            for val_idx in range(num_val):
                val_images = val_data[val_idx].to(device)
                val_images = model(val_images)
                val_gt_images = val_gt_data[val_idx].to(device)
          
                loss_val += loss_cls(val_images, val_gt_images) / num_val

            val_psnr = 10. * torch.log10(1.0 / loss_val)
        model.train()
        end_t = time.time()

        print(
            "(Epoch {} / {}) train loss: {:.6f} val PSNR: {:.6f} time per epoch: {:.1f}s".format(
                i, num_epochs, loss.item(), val_psnr.item(), end_t - start_t
            )
        )

        #lr_scheduler.step()

    # plot the training losses
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training loss history")
    plt.show()
    return model, loss_history

def get_eval_img_set5(path, scaling_factor):
    if scaling_factor == 2:
        filenames = [f'img_00{i}_SRF_2_' for i in range(1, 6)]
    elif scaling_factor ==3:
        filenames = [f'img_00{i}_SRF_3_' for i in range(1, 6)]
    batch_size = len(filenames)
    eval_in = []
    eval_out = []
    for i, filename in enumerate(filenames):
        file_in = filename + 'LR.png'
        file_out = filename + 'HR.png'
        # file in 
        img = Image.open(os.path.join(path, file_in))
        if torch.tensor(np.array(img)).dim() < 3:
            gray = torch.tensor(np.array(img))
            img = torch.stack([gray,gray,gray], dim=2)
            img = img.permute(2, 0, 1)
        else:
            img = torch.tensor(np.array(img)).permute(2, 0, 1)
        eval_in.append(img)
        # file out
        
        img = Image.open(os.path.join(path, file_out))
        if torch.tensor(np.array(img)).dim() < 3:
            gray = torch.tensor(np.array(img))
            img = torch.stack([gray,gray,gray], dim=2)
            img = img.permute(2, 0, 1)
        else:
            img = torch.tensor(np.array(img)).permute(2, 0, 1)
        eval_out.append(img)
        
    return eval_in, eval_out

def get_eval_img_set14(path, scaling_factor):
    if scaling_factor == 2:
        filenames = [f'img_00{i}_SRF_2_' for i in range(1, 10)] + [f'img_01{i}_SRF_2_' for i in range(0, 5)]
    elif scaling_factor == 3:
        filenames = [f'img_00{i}_SRF_3_' for i in range(1, 10)] + [f'img_01{i}_SRF_3_' for i in range(0, 5)]
    batch_size = len(filenames)
    eval_in = []
    eval_out = []
    for i, filename in enumerate(filenames):
        file_in = filename + 'LR.png'
        file_out = filename + 'HR.png'
        # file in 
        img = Image.open(os.path.join(path, file_in))
        if torch.tensor(np.array(img)).dim() < 3:
            gray = torch.tensor(np.array(img))
            img = torch.stack([gray,gray,gray], dim=2)
            img = img.permute(2, 0, 1)
        else:
            img = torch.tensor(np.array(img)).permute(2, 0, 1)
        eval_in.append(img)
        # file out
        
        img = Image.open(os.path.join(path, file_out))
        if torch.tensor(np.array(img)).dim() < 3:
            gray = torch.tensor(np.array(img))
            img = torch.stack([gray,gray,gray], dim=2)
            img = img.permute(2, 0, 1)
        else:
            img = torch.tensor(np.array(img)).permute(2, 0, 1)
        eval_out.append(img)
        
    return eval_in, eval_out


def inference(model, testing_data, dtype, mode, scale_factor):
    # case 1: Y only
    # input : batch of RGB low-resolution images
    # to YCbCr
    testing_data = testing_data.to(dtype)
  
    test_data_input_ycbcr_np = skimage.color.rgb2ycbcr(testing_data.permute(0,2,3,1).contiguous().cpu())
    # get Y channel only
    # and normalize data
    test_data_input_y = torch.from_numpy(test_data_input_ycbcr_np).permute(0,3,1,2).contiguous()[:,0:1,:,:] / 235.0
    test_data_input_y = test_data_input_y.to(model.device)
    if mode == 'fsrcnn':
      pred_y = model(test_data_input_y).cpu()
    elif mode == 'bilinear':
      pred_y = torch.nn.functional.interpolate(test_data_input_y, scale_factor=scale_factor, mode='bilinear', recompute_scale_factor=True, align_corners=False).cpu()
    elif mode == 'bicubic':
      pred_y = torch.nn.functional.interpolate(test_data_input_y, scale_factor=scale_factor, mode='bicubic', recompute_scale_factor=True, align_corners=False).cpu()
      

    pred_y = pred_y.clamp(16.0 / 235.0, 1.0) * 235.0
    
    # combine with the original CbCr channel but after bicubic up-sampling
    # get CbCr channel with high-resolutino image
    test_data_input_cbcr = torch.from_numpy(test_data_input_ycbcr_np).permute(0,3,1,2).contiguous()[:,1:3,:,:]
    pred_cbcr = torch.nn.functional.interpolate(test_data_input_cbcr, scale_factor=scale_factor, mode='bicubic', recompute_scale_factor=True, align_corners=False).cpu()
    pred_cbcr = pred_cbcr.clamp(16.0, 240.0)
    
    rebuild_data_ycbcr = torch.cat((pred_y, pred_cbcr),dim=1)
    rebuild_data_rgb_np = skimage.color.ycbcr2rgb(rebuild_data_ycbcr.permute(0,2,3,1).contiguous().cpu().detach().numpy())
    rebuild_data = torch.from_numpy(rebuild_data_rgb_np).permute(0,3,1,2).contiguous()
    

    return pred_y, rebuild_data.squeeze(0)

def eval_model(pred, gt):
    gt, pred = gt.cpu().detach().numpy(), pred.cpu().detach().numpy()
    psnr = metrics.peak_signal_noise_ratio(gt, pred, data_range=255)
    ssim = metrics.structural_similarity(gt, pred, data_range=255)
    return psnr, ssim


