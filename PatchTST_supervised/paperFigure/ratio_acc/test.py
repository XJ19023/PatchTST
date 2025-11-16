import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()

# 左边 y 轴
line1, = ax1.plot([1, 2, 3], [1, 4, 9], "b-", label="Left Y")
ax1.set_ylabel("Left Y")

# 右边 y 轴
ax2 = ax1.twinx()
line2, = ax2.plot([1, 2, 3], [1, 2, 3], "r--", label="Right Y")
ax2.set_ylabel("Right Y")

# 合并两个轴的 legend
lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="best")

plt.show()



plt.savefig(f'aaa.png')
# plt.savefig('energy.png', bbox_inches='tight')
plt.close()