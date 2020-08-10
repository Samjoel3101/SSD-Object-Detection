from fastai2.callback.hook import hook_output
from .layers import ResBlock, OutConv
from .utils import model_idx_for_gridsize 
from fastai2.vision.all import xresnet34


class SSD_Model(nn.Module):

  def __init__(self, grids = [4, 2, 1], backbone_grids = [28, 14, 7], 
               backbone = xresnet34, pretrained = True,
               num_categories = None, anchors_per_cell = 9, num_id_convs = 1,
               is_cut = True,
               ):
    super().__init__()

    self.num_categories = num_categories 
    self.anchors_per_cell = anchors_per_cell
    self.grids =  grids
    self.backbone_grids = backbone_grids
    self.total_grids = self.backbone_grids + self.grids
    self.sizes = [[grid, grid] for grid in self.total_grids]

    if not is_cut:
      self.backbone = backbone(pretrained = pretrained)
      self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
    else:
      self.backbone = backbone 

    outputs = self._register_hooks()
    self.outputs = outputs
    self.backbone_convs = self._hook_convs(self.outputs, num_id_convs = num_id_convs)
    self.conv_blocks, self.out_blocks = self._setup_out_convs() 

    if torch.cuda.is_available(): self.device = torch.device('cuda')
    else: self.device = torch.device('cpu')


  def forward(self, x):

    classification_results = []
    bbox_results = []
    out = self.backbone(x)
    for resblock, outblock in zip(self.conv_blocks, self.out_blocks):
      out = resblock(out)
      clas_out, bbox_out = outblock(out)
      classification_results.append(clas_out); bbox_results.append(bbox_out)

    for idx, module_list in enumerate(self.backbone_convs):
      current_hook = self.outputs[idx].stored
      clas_out, bbox_out = module_list(current_hook)
      classification_results.append(clas_out); bbox_results.append(bbox_out)
    return [torch.cat(classification_results, dim = 1), torch.cat(bbox_results, dim = 1), self.sizes]
  
  def _register_hooks(self):
    outputs = []
    self.backbone_idxs = model_idx_for_gridsize(self.backbone)
    for grid_size, index in self.backbone_idxs:
      outputs.append(hook_output(self.backbone[index], detach = False))
    return outputs
  
  def _module_list_for_hooks(self, hooks):
    backbone_convs = []
    for idx, hook in enumerate(hooks):
      backbone_convs.append(nn.Sequential())
    return backbone_convs
  
  def _hook_convs(self, hooks, num_id_convs = 1):
    backbone_convs = self._module_list_for_hooks(hooks)
    dummy_out = self.backbone(self._dummy_batch)
    for idx, module_list in enumerate(backbone_convs):
      current_hook_out = hooks[idx].stored
      _, in_channels, grid, grid = current_hook_out.shape
      nf = 256
      for i in range(num_id_convs): 
        module_list.add_module(f'idconv{i}', ResBlock(in_channels, nf, stride = 1))
        if nf > 64: nf /= 2
      module_list.add_module(f'out_conv{i}', OutConv(int(nf*2), num_categories = self.num_categories,
                                                     num_anchorbxs = self.anchors_per_cell))
    return nn.ModuleList(backbone_convs)

  @property
  def _dummy_batch(self, input_shape = 224): return torch.rand((2, 3, input_shape, input_shape))

  @property 
  def _backbone_final_channel(self):
    _, channel, grid, grid = self.backbone(self._dummy_batch).shape
    return channel 
  
  def _setup_out_convs(self):
    conv_blocks = []; nf = 256; out_blocks = []
    for idx, grid_size in enumerate(self.grids):
      conv_blocks.append(ResBlock(self._backbone_final_channel if idx == 0 else nf, nf, stride = 2,
                                       downsampler = None if idx == 0 else nn.AvgPool2d(2, 2, ceil_mode= True)))
      out_blocks.append(OutConv(nf, num_categories = self.num_categories, num_anchorbxs = self.anchors_per_cell))
    
    return nn.ModuleList(conv_blocks), nn.ModuleList(out_blocks)