import torch 
from torch import nn 
import torch.nn.functional as F 

def one_hot(labels, num_classes):
  return torch.eye(num_classes)[labels.data.cpu()]

class BCE_Loss(nn.Module):
  def __init__(self, num_classes):
    super().__init__()
    self.num_classes = num_classes
  
  def forward(self, pred, targ):
    # pdb.set_trace()
    target = one_hot(targ, self.num_classes)
    target = torch.Tensor(target[:, 1:].contiguous()).cuda()
    pred = pred[:, 1:]
    weight = self.get_weights(pred, target)
    return F.binary_cross_entropy_with_logits(pred, target, weight, size_average = False)/(self.num_classes - 1)
  
  def get_weights(self, pred, target): return None