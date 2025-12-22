import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches
from matplotlib.ticker import PercentFormatter
import torch

tensor = torch.tensor([0.1486522 ,
0.1492039 ,
0.15035428,
0.14929476,])

print(tensor[0] * 1.01)
