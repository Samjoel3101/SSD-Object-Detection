import numpy as np 
import ipywidgets as widgets 
import matplotlib.patches as pat
from ipywidgets import HBox, VBox 


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


class Scheduler():
  def __init__(self, learn:Learner, num_groups = None, figsize = (8, 8)):
    self.figsize = figsize
    self.learn = learn

    if learn.opt is None:
      learn.create_opt()
    self.length = len(learn.opt.param_groups)

    self.setup_widgets()
    self.attr_values = [SchedNo(1e-3, 1e-3)]*self.length

    self.first_cycle_scheduler = self.scheduler_types['Cos']
    self.second_cycle_scheduler = self.scheduler_types['Cos']

  def learner(self):  

    # Removing Existing Param schedulers
    for cb in self.learn.cbs:
      if 'param_scheduler' in cb.name:
        self.learn.remove_cb(cb)

    # Setting a new Param Scheduler with the set parameters 
    self.sched = {'lr': self.attr_values}
    scheduler = XParamScheduler(self.sched)
    self.learn.add_cb(scheduler) 

    return self.learn

  def setup_widgets(self):
    # First Cycle widgets
    self.start_widgets_1st_cycle = []
    self.end_widgets_1st_cycle = []

    # Second Cycle widgets
    self.start_widgets_2nd_cycle = []
    self.end_widgets_2nd_cycle = []

    #Splitters
    self.splits = []

    # Radio Button to toggle between different Layer Groups
    self.idxs = widgets.RadioButtons(options = [f'Layer Group {idx}' 
                                                for idx in range(self.length)])
    
    self.pcts = torch.linspace(0., 1., 50) # Dummy pcts 

    # Assigning scales to each slider value ## Coz sliders just output between 0 and 1 efficiently 
    # eg. for a slider value of 0.9 its multiplied by 1e-3 or user's choice so it becomes 0.9 * 1e-3
    self.scales = [[widgets.Text(value = '1e-3', continuous_update = False) for i in range(4)] 
                                                                  for j in range(self.length)]

    # Buttons for declaring the type of scheduler for each cycle  
    self.first_cycle_btns = [widgets.Button(description = f'{name}') 
                              for name in self.scheduler_types.keys()]
    self.second_cycle_btns = [widgets.Button(description = f'{name}')
                              for name in self.scheduler_types.keys()]
    self._setup_btns() # Setting Up each btn's on_click attribute to execute a callback func

    for idx in range(self.length):
      self.start_widgets_1st_cycle.append(self._create_float_slider(description = f'Layer Group {idx}'))
      self.end_widgets_1st_cycle.append(self._create_float_slider( description = f'Layer Group {idx}'))
      self.start_widgets_2nd_cycle.append(self._create_float_slider(description = f'Layer Group {idx}'))
      self.end_widgets_2nd_cycle.append(self._create_float_slider( description = f'Layer Group {idx}'))
      self.splits.append(self._create_float_slider(value = 0.3, min = 0.1, max = 1., 
                                                   step = 0.05, description = f'Split {idx}',
                                                  ))

  def _create_float_slider(self, value = 2., max = 1., min = 0., step = 0.01, description = None, **kwargs):
    widget = widgets.FloatSlider( value = value,
                                  min = min, max = max,
                                  step = step, description = description,
                                  continuous_update = True, **kwargs) 
    return widget 
  
  def schedule(self):
    widgets.interact(self.plot, idx = self.idxs)

  def _store_values(self, idx, scheduler):
    self.attr_values[idx] = scheduler
  
  @property
  def scheduler_types(self):
    return {'Lin': SchedLin, 'Cos': SchedCos, 'Exp': SchedExp, 'No_Schedule': SchedNo}

  def _setup_btns(self):
    for first_cyc_btn, second_cyc_btn in zip(self.first_cycle_btns, self.second_cycle_btns):
      first_cyc_btn.on_click(self.set_first_cycle_scheduler)
      second_cyc_btn.on_click(self.set_second_cycle_scheduler)

  def set_first_cycle_scheduler(self, button): 
    print('First Cycle', button.description)
    self.first_cycle_scheduler = self.scheduler_types[button.description]
  
  def set_second_cycle_scheduler(self, button):
    print('Second Cycle', button.description)
    self.second_cycle_scheduler = self.scheduler_types[button.description]
  
  def plot(self, idx):
    idx = int(idx[-1])
    
    def _executer(scale_1, scale_2, scale_3, scale_4, start_1, end_1, start_2, end_2, split):
      scale_1, scale_2, scale_3, scale_4 = float(scale_1), float(scale_2), float(scale_3), float(scale_4)
      print(start_1*scale_1, end_1*scale_2, start_2*scale_3, end_2*scale_4, split, 1-split)
      scheduler = combine_scheds([split, 1-split], [self.first_cycle_scheduler(start_1*scale_1, end_1*scale_2), 
                                                    self.second_cycle_scheduler(start_2*scale_3, end_2*scale_4)])
      self._store_values(idx, scheduler)
      plt.plot(self.pcts, [scheduler(o) for o in self.pcts])
 
    w = widgets.interactive(_executer, 
                            start_1 = self.start_widgets_1st_cycle[idx], end_1 = self.end_widgets_1st_cycle[idx],
                            start_2 = self.start_widgets_2nd_cycle[idx], end_2 = self.end_widgets_2nd_cycle[idx],
                            scale_1 = self.scales[idx][0], scale_2 = self.scales[idx][1], 
                            scale_3 = self.scales[idx][2], scale_4 = self.scales[idx][3],
                            split = self.splits[idx])
    
    output = w.children[-1]
    output.layout.height = '350px'
    display(  VBox([
                    widgets.HTML(value = r"<div  style = 'padding-left: 125px'> <h1> First Cycle </h1> </div>"),
                    HBox(self.first_cycle_btns),
                    VBox([HBox([w.children[4],  w.children[5]]), 
                          HBox([w.children[0], w.children[1]])]),  
                    w.children[8],
                    widgets.HTML(value = r"<div  style = 'padding-left: 125px'> <h1> Second Cycle </h1> </div>"),
                    HBox(self.second_cycle_btns),
                    VBox([HBox([w.children[6],  w.children[7]]), 
                          HBox([w.children[2], w.children[3]])]),
                    output
                    ]))   
    




