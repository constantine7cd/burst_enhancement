import torch
import numpy as np

from .network import DenoiseNet
from .torch_utils import load_checkpoint

PATH_TO_WEIGHTS = "/Users/Konstantin/hdr_plus_plus/src/denoising/model_weights/dnd_raw.pth"

model_restoration = DenoiseNet()
load_checkpoint(model_restoration, PATH_TO_WEIGHTS)

model_restoration.cpu()
model_restoration.eval()


def denoiser(raw_noisy, variance):
    raw_noisy = torch.Tensor(raw_noisy).unsqueeze(0).permute(0,3,1,2).cpu()
    variance = torch.Tensor(variance).unsqueeze(0).permute(0,3,1,2).cpu()
 
    with torch.no_grad():          
        raw_restored = model_restoration(raw_noisy, variance)

    raw_restored = torch.clamp(raw_restored,0,1)                
    
    raw_restored = raw_restored.permute(0, 2, 3, 1).cpu().detach().numpy()[0]
    return raw_restored