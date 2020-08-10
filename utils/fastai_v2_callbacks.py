from fastai2.learner import save_model
from fastai2.callback.core import Callback 
from fastai2.callback import ParamScheduler

# Save Model at a pct if it takes more than six hours to train 
class PctSaveCallback(Callback):
  def __init__(self, path):
    super().__init__()
    self.path = path 
  
  def after_batch(self):
    if self.pct_train > 0.5 and not self.path.exists():
      save_model(self.path, model = self.model, opt = self.opt)
  
  def after_epoch(self):
    save_model(self.path, model = self.model, opt = self.opt)

# Lr Recorder for each Parameter Group
class Lr_Recorder(Callback):
  def __init__(self, figsize = (8, 8)):
    self.figsize = figsize

  def begin_fit(self):
    self.length = len(self.opt.param_groups)
    self.losses = []
    self.train_pct = []
    self.lrs = [[] for i in range(self.length)]

  def after_batch(self):
    self.losses.append(self.loss); self.train_pct.append(self.pct_train)
    self.losses.append(self.loss)
    for idx, p_grp in enumerate(self.opt.param_groups):
      self.lrs[idx].append(p_grp['lr'])

  def plot(self):
    _, axs = plt.subplots(self.length//2, 2, figsize = self.figsize)
    axs = axs.flatten()
    for idx, lrs in enumerate(self.lrs):
      axs[idx].plot(self.train_pct, self.lrs[idx])
      axs[idx].set_title(f'Layer Group {idx + 1}')
    plt.show()

# A ParamScheduler which deals with multiple scheduler funcs for different layer groups
class XParamScheduler(ParamScheduler):
  def begin_batch(self):
    num_schedulers = len(list(self.scheds.values())[0])
    num_param_groups = len(self.opt.param_groups)
    if num_schedulers < num_param_groups: 
      super().begin_fit()
    else:
      for p, func_list in self.scheds.items():
        value_list = [f(self.pct_train) for f in func_list]
        self.opt.set_hyper(p, value_list)