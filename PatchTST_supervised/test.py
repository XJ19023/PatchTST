import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches
from matplotlib.ticker import PercentFormatter
import torch

tensor = torch.tensor([0.31933215,
0.31935361,
0.32020786,
0.31946877,
0.3192693, ])

print(tensor[0] * 1.001)
