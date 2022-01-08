"""
https://github.com/huangzh13/StyleGAN.pytorch/blob/master/utils/__init__.py
"""

from .logger import make_logger
from .copy import list_dir_recursively_with_ignore, copy_files_and_create_dirs
import torch
import matplotlib.pyplot as plt
import PIL
import math

# Load fewer layers of pre-trained models if possible
def load(model, cpk_file, map_location= 'cpu'):
    pretrained_dict = torch.load(cpk_file, map_location= 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

def visualize_tensor(fakes):
    fig = plt.figure(figsize= (15, 15))
    num_sample = fakes.size(0)
    edge = int(math.sqrt(num_sample))
    
    numpy_images = fakes.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
    for i, img in enumerate(numpy_images):
        plt.subplot(edge, edge, i + 1)
        plt.imshow(PIL.Image.fromarray(img))
        plt.axis('off')
    # plt.show()
    return fig