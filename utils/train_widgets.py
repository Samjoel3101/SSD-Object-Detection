import ipywidgets as widgets 
from ipywidgets import HBox, VBox 

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
    