# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
figure_path = '/Users/hualin/ENV/workspace/lifecycle.wiki/paper_writing/figure/'
filename = 'plc.txt'
glb_fontweight='light'

plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体，则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

# label在图示(legend)中显示。若为数学公式，则最好在字符串前后添加"$"符号
# color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
# 线型：-  --   -.  :    ,
# marker：.  ,   o   v    <    *    +    1
plt.figure(figsize=(6, 3.0))
plt.grid(linestyle="--")  # 设置背景网格线为虚线
ax = plt.gca()
ax.spines['top'].set_visible(False)  # 去掉上边框
ax.spines['right'].set_visible(False)  # 去掉右边框

data = np.loadtxt(filename)

# max_id = int(data[-1, 0])
max_id = 10
x_max = int(np.max(data[:, 2], axis=0) + 1)

for k in range(max_id):
    idx = np.where(data[:,0]==k)[0]
    plt.plot(data[idx, 2], data[idx, 3], color='0.8', linewidth=0.5)
idx = np.where(data[:, 0]== 1)[0]
plt.plot(data[idx, 2], data[idx, 3], color='0.2', linewidth=1.0, label='Typical Lifecycle')
# plt.plot(x, A, color="black", label="A algorithm", linewidth=1.5)


# group_labels = ['dataset1', 'dataset2', 'dataset3', 'dataset4', 'dataset5', ' dataset6', 'dataset7', 'dataset8',
#                 'dataset9', 'dataset10']  # x轴刻度的标识
xspace = range(0, x_max, 20)
yspace = np.arange(0.05, 0.16, 0.03)
plt.xticks(xspace, fontsize=12, fontweight=glb_fontweight)  # 默认字体大小为10
plt.yticks(yspace, fontsize=12, fontweight=glb_fontweight)
# plt.title("example", fontsize=12, fontweight=glb_fontweight)  # 默认字体大小为12
plt.xlabel("Time", fontsize=13, fontweight=glb_fontweight)
plt.ylabel("CTR", fontsize=13, fontweight=glb_fontweight)
plt.xlim(0, x_max)  # 设置x轴的范围
plt.ylim(0.05, 0.15)

# plt.legend()          #显示各曲线的图例
plt.legend(loc=0, numpoints=1)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=12, fontweight=glb_fontweight)  # 设置图例字体的大小和粗细
plt.tight_layout()

plt.savefig(figure_path+'lifecycle.pdf', format='pdf')  # 建议保存为svg格式，再用inkscape转为矢量图emf后插入word中
plt.show()