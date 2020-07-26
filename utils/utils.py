import torch 
from torch import nn 
import torch.nn.functional as F

def get_specific_grid_size(model, image_size = [1, 3, 224, 224]):
  rand_img = torch.rand(*image_size)
  real_img = rand_img
  downsample_shapes = []
  model_layers = model
  for idx, l in enumerate(model_layers.children()):
    try:
      rand_img = l(rand_img)
      if (rand_img.shape[2] != real_img.shape[2]) and (rand_img.shape[2] not in downsample_shapes):
        downsample_shapes.append(rand_img.shape[2])
    except Exception: continue
  print('Available Grid Sizes', downsample_shapes)

def model_idx_for_gridsize(model, grid_sizes = [28, 14, 7]):
  rand_img = torch.rand(1, 3, 224, 224) 
  model_idx = []
  for idx, layer in enumerate(model.children()):
    rand_img = layer(rand_img)
    try:
      if rand_img.shape[2] in grid_sizes:
        model_idx.append((rand_img.shape[2], idx))
    except Exception: break
  return model_idx