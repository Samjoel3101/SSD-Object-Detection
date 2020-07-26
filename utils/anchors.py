import numpy as np 
import ipywidgets as widgets 
import matplotlib.patches as pat


class Anchor_Visualizer:
  """
  widgets = the attributes for which widgets are created 
      -> default = ['anchor offset']
  """

  def __init__(self, 
               grid_size = 4, 
               zooms = [0.7, 1., 1.3], 
               ratios = [[1., 1.], [0.5, 1.], [1., 0.5]],
               anc_func = None  
                ):
    
    self.zooms = zooms 
    self.grid_size = grid_size
    self.ratios = ratios
    self.scale
    if anc_func is not None:
      self.anc_func = anc_func 
    else: self.anc_func = self.get_anchors   
    self.setup_widgets()

  def setup_widgets(self):
    self.offset_widget = widgets.FloatSlider(min = 0.1, max = 1.5, step = 0.05, value = 0.5)
  
  def get_anchors(self, offset_widget):
    self.anc_scales = [(anz*i, anz*j) for anz in self.zooms for i, j in self.ratios]
    self.grid_offset = offset_widget/self.grid_size
    range_values = np.linspace(self.grid_offset, 1-self.grid_offset, self.grid_size)
    self.anc_xs = np.repeat(range_values, self.grid_size)
    self.anc_ys = np.tile(range_values, self.grid_size)
    self.anc_ctrs = np.repeat(np.stack([self.anc_xs, self.anc_ys], axis = 1), self.num_boxes_per_grid, axis = 0)
    self.anc_sz = np.array([[w/self.grid_size, h/self.grid_size] for i in range(self.grid_size**2) 
                                                          for w, h in self.anc_scales])
    anchors = np.concatenate([self.anc_ctrs, self.anc_sz], axis = 1)
    return anchors

  @property
  def num_boxes_per_grid(self): return len(self.anc_scales)

  @property
  def total_boxes(self): return self.anchors.shape[0]

  @property 
  def scale(self): return self.grid_size*100 

  @property 
  def blank_image(self): return np.full((self.scale, self.scale, 3), 255, np.uint8)/255.

  def tfm_coords(self, coords):
    x, y, width, height = coords
    x = x - width/2
    y = y - height/2 
    return np.array([x, y, width, height])

  def plot_anchors(self, offset_widget):
    fig, ax = plt.subplots(1, figsize = (15, 15))
    ax.clear()
    anchors = self.anc_func(offset_widget)
    ax.imshow(self.blank_image)

    for idx, coords in enumerate(anchors):
      x_ctr, y_ctr = coords[:2]*self.scale
      ctr_patch = pat.Circle((x_ctr, y_ctr), 0.009*self.scale)
      x, y, width, height = self.tfm_coords(coords)*self.scale
      rect_patch = pat.Rectangle((x, y), width, height, linewidth = 1, edgecolor = 'r', facecolor = 'none')
      ax.add_patch(ctr_patch)
      ax.add_patch(rect_patch)

  def plot(self):
     return widgets.interactive(self.plot_anchors, offset_widget = self.offset_widget)

def plot_anchors(bbs, size):
  scale = size*100
  image = np.full((scale, scale, 3), 255, np.uint8)/255.
  fig, ax = plt.subplots(1, figsize = (10, 10))
  ax.imshow(image)
  for idx, bb in enumerate(bbs):
    x, y, x2, y2 = bb*scale 
    width, height = x2-x, y - y2
    patch = pat.Rectangle((x, y), width, height, edgecolor = 'r', facecolor = 'none')
    ax.add_patch(patch)
  plt.show()


