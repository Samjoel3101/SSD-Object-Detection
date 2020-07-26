import torch 
from torch import nn 
import torch.nn.functional as F 

def conv_layer(ni, nf, k, stride, act = True, zero_bn = False):
  layers = [nn.Conv2d(ni, nf, k, stride = stride, padding = k//2, bias = False)]
  bn = nn.BatchNorm2d(nf)
  if zero_bn:
    bn.weight.data.zero_()
  layers.append(bn)
  if act:
    layers.append(nn.ReLU())
  return nn.Sequential(*layers)  

class MergeLayer(nn.Module):
  def __call__(self, x1, x2, dense = False):
    if dense:
      return torch.cat([x1, x2], dim = 1)
    else:
      return x1+x2

class ResBlock(nn.Module):

  def __init__(self, ni, nf, stride, downsampler = None):
    super().__init__()
    self.downsampler = downsampler
    self.stride = stride  
    nh = nf//2
    self.conv1 = conv_layer(ni, nh, 1, stride = 1)
    self.conv2 = conv_layer(nh, nh, 3, stride = stride)
    self.conv3 = conv_layer(nh, nf, 1, stride = 1, act = False, zero_bn = True)

    self.merge_layer = MergeLayer()

    if self.downsampler is None:
      self.downsampler = nn.AvgPool2d(2, 2, padding = 1)
    # Downsampling Path 
    self.conv4 = conv_layer(ni, nf, 1, stride = 1, act = False)
    self.act_fn = nn.ReLU()

  def forward(self, x):
    residual = x
    out = self.conv3(self.conv2(self.conv1(x)))
    if self.stride > 1:
      residual = self.downsampler(residual)
    residual = self.conv4(residual)
    return self.act_fn(self.merge_layer(residual, out))

def flatten_conv(x, num_anchors_per_cell):
  bs, nf, grid, grid = x.shape
  return x.view(bs, -1, nf//num_anchors_per_cell)

class OutConv(nn.Module):
  def __init__(self, ni, k = 1, stride = 1, num_categories = None, num_anchorbxs = 9):
    super().__init__()
    self.num_anchorbxs = num_anchorbxs 
    self.classifier_head = nn.Conv2d(ni, int((num_categories)*num_anchorbxs), k, stride = 1, padding = k//2) 
    self.regressor_head = nn.Conv2d(ni, int(4*num_anchorbxs), k, stride = 1, padding = k//2)
  
  def forward(self, x):
    classification_results = self.classifier_head(x)
    bboxes = self.regressor_head(x)
    return [flatten_conv(classification_results, self.num_anchorbxs), 
            flatten_conv(bboxes, self.num_anchorbxs)]