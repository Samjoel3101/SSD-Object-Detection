from fastai import * 
from fastai.vision import * 
from fastai.callbacks.hooks import hook_output
from .layers import * 
from .utils import *
 
class SSD_Model(nn.Module):

  def __init__(self, grids = [4, 2, 1], backbone_grids = [28, 14, 7], 
               backbone = models.xresnet34, pretrained = True,
               num_categories = None, anchors_per_cell = 9
               ):
    super().__init__()

    self.num_categories = num_categories 
    self.anchors_per_cell = anchors_per_cell  
    self.backbone = nn.Sequential(*list(backbone(pretrained).children())[:-2])
    self.backbone_grids = backbone_grids 
    self.grids =  grids
    self.hook_convs()    
    self.setup_out_convs()

    if torch.cuda.is_available(): self.device = torch.device('cuda')
    else: self.device = torch.device('cpu')

    self.setup_device() 

  def forward(self, x):

    classification_results = []
    bbox_results = []
    # pdb.set_trace()
    out = self.backbone(x)
    for resblock, outblock in zip(self.conv_blocks, self.out_blocks):
      out = resblock(out)
      clas_out, bbox_out = outblock(out)
      classification_results.append(clas_out); bbox_results.append(bbox_out)

    for idx, module_list in enumerate(self.backbone_convs):
      current_hook = self.outputs[idx].stored
      clas_out, bbox_out = module_list(current_hook)
      classification_results.append(clas_out); bbox_results.append(bbox_out)
    return torch.cat(classification_results, dim = 1), torch.cat(bbox_results, dim = 1)
  
  def register_hooks(self):
    self.outputs = []
    self.backbone_idxs = model_idx_for_gridsize(self.backbone)
    for grid_size, index in self.backbone_idxs:
      self.outputs.append(hook_output(self.backbone[index], detach = False))
  
  def module_list_for_hooks(self):
    self.register_hooks()
    self.backbone_convs = []
    for idx, hook in enumerate(self.outputs):
      self.backbone_convs.append(nn.Sequential())
  
  def hook_convs(self, num_id_convs = 1):
    self.module_list_for_hooks()
    dummy_out = self.backbone(self.dummy_batch)
    for idx, module_list in enumerate(self.backbone_convs):
      current_hook_out = self.outputs[idx].stored
      _, in_channels, grid, grid = current_hook_out.shape
      nf = 256
      for i in range(num_id_convs): 
        module_list.add_module(f'idconv{i}', ResBlock(in_channels, nf, stride = 1))
        if nf > 64: nf /= 2
      module_list.add_module(f'out_conv{i}', OutConv(int(nf*2), num_categories = self.num_categories,
                                                     num_anchorbxs = self.anchors_per_cell))

  @property
  def dummy_batch(self, input_shape = 224): return torch.rand((2, 3, input_shape, input_shape))

  @property 
  def backbone_final_channel(self):
    _, channel, grid, grid = self.backbone(self.dummy_batch).shape
    return channel 
  
  def setup_out_convs(self):
    self.conv_blocks = []; nf = 256; self.out_blocks = []
    for idx, grid_size in enumerate(self.grids):
      self.conv_blocks.append(ResBlock(self.backbone_final_channel if idx == 0 else nf, nf, stride = 2,
                                       downsampler = None if idx == 0 else nn.AvgPool2d(2, 2, ceil_mode= True)))
      self.out_blocks.append(OutConv(nf, num_categories = self.num_categories, num_anchorbxs = self.anchors_per_cell))
  
  def setup_device(self):
    self.backbone = self.backbone.to(self.device)
    self.conv_blocks = [blk.to(self.device) for blk in self.conv_blocks]
    self.out_blocks = [blk.to(self.device) for blk in self.out_blocks]
    self.backbone_convs = [blk.to(self.device) for blk in self.backbone_convs]

  def __repr__(self):
    super().__repr__()
    text = self.backbone.__repr__()
    for idx, grid_size in enumerate(self.backbone_grids):
      text += f'Grid Size: {grid_size} \n'
      text += str(self.backbone_convs[idx]) + '\n'

    for idx, (resblock, outblock) in enumerate(zip(self.conv_blocks, self.out_blocks)):
      text += f'Grid Size: {self.grids[idx]} \n'
      text += str(resblock) + '\n'
      text += str(outblock) + '\n'
    
    return text

  def setup_optimizer(self):
    self.optim = OptimWrapper(torch.optim.Adam([
      {'params': self.backbone.parameters(), 'lr': 1e-6},
      {'params': nn.ModuleList(self.out_blocks).parameters(), 'lr': 1e-3},
      {'params': nn.ModuleList(self.conv_blocks).parameters(), 'lr':1e-2},
      {'params': nn.ModuleList(self.backbone_convs).parameters(), 'lr':1e-3}                                  
    ]))