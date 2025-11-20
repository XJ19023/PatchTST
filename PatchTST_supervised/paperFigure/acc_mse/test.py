import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from matplotlib.text import OffsetFrom
fig, ax1 = plt.subplots(1, 1)

# 第一个注释：去掉 bbox
an1 = ax1.annotate(' ', xy=(.5, .7), xycoords='data',
                   ha="center", va="center")

an2 = ax1.annotate(' ', xy=(.5, .5), xycoords=an1,
                   xytext=(.5, .3), textcoords='data',
                   ha="center", va="center",
                   arrowprops=dict(
                       patchB=None,   # 不再依赖 an1 的 bbox
                       connectionstyle="arc3,rad=0.2",
                        arrowstyle="->",
                        edgecolor="#d43f3b",   # 线段颜色
                        facecolor="#d43f3b",  # 箭头填充颜色
                        lw=2
    ))

# an1.draggable()
# an2.draggable()

# an3 = ax1.annotate('', xy=(.5, .5), xycoords=an2,
#                    xytext=(.5, .5), textcoords=an1,
#                    ha="center", va="center",
#                    bbox=bbox_args,
#                    arrowprops=dict(patchA=an1.get_bbox_patch(),
#                                    patchB=an2.get_bbox_patch(),
#                                    connectionstyle="arc3,rad=0.2",
#                                    **arrow_args))

# Finally we'll show off some more complex annotation and placement

# text = ax2.annotate('xy=(0, 1)\nxycoords=("data", "axes fraction")',
#                     xy=(0, 1), xycoords=("data", 'axes fraction'),
#                     xytext=(0, -20), textcoords='offset points',
#                     ha="center", va="top",
#                     bbox=bbox_args,
#                     arrowprops=arrow_args)

# ax2.annotate('xy=(0.5, 0)\nxycoords=artist',
#              xy=(0.5, 0.), xycoords=text,
#              xytext=(0, -20), textcoords='offset points',
#              ha="center", va="top",
#              bbox=bbox_args,
#              arrowprops=arrow_args)

# ax2.annotate('xy=(0.8, 0.5)\nxycoords=ax1.transData',
#              xy=(0.8, 0.5), xycoords=ax1.transData,
#              xytext=(10, 10),
#              textcoords=OffsetFrom(ax2.bbox, (0, 0), "points"),
#              ha="left", va="bottom",
#              bbox=bbox_args,
#              arrowprops=arrow_args)


plt.savefig(f'aaa.png')
plt.show()