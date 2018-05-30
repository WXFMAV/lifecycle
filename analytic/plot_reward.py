# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sys 
#figure_path = '/Users/hualin/ENV/workspace/lifecycle.wiki/paper_writing/figure/'
figurepath = './fig/'
datpath = './dat/'
filename = datpath + 'reward.txt'
figfile = figurepath + 'reward'

if len(sys.argv) == 5:
    datpath = sys.argv[1]
    figurepath = sys.argv[2]
    filename = datpath + sys.argv[3]
    figfile = figurepath +sys.argv[4]

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
x_min = int(np.min(data[:, 0], axis = 0))
x_max = int(np.max(data[:, 0], axis = 0) + 1)
y_min = float(np.min(data[:, 1], axis = 0))
y_max = float(np.max(data[:, 1], axis = 0)) + 0.5

sample_times = 1000
sample_step = int((x_max - x_min)/sample_times)

sep = []
for k in range(0, sample_times - 1):
    sep.append((k) * sample_step)
sep.append(x_max)

res = np.zeros(shape=(len(sep) - 1, 5))

for k in range(0, len(sep) - 1):
    res[k, 0] = sep[k]
    res[k, 1] = np.mean(data[sep[k]:sep[k+1], 1])
    res[k, 2] = np.std(data[sep[k]:sep[k+1], 1])
    res[k, 3] = np.min(data[sep[k]:sep[k+1], 1])
    res[k, 4] = np.max(data[sep[k]:sep[k+1], 1])
plt.plot(res[:, 0], res[:, 1], color='0.2', linewidth=1.0, label='reward')
# plt.fill_between(res[:, 0], res[:, 1] - res[:, 2], res[:, 1] + res[:, 2], color='0.8',alpha=0.25)
plt.fill_between(res[:, 0], res[:, 3], res[:, 4], color='0.6',alpha=0.25)
# plt.plot(x, A, color="black", label="A algorithm", linewidth=1.5)

# group_labels = ['dataset1', 'dataset2', 'dataset3', 'dataset4', 'dataset5', ' dataset6', 'dataset7', 'dataset8',
#                  'dataset9', 'dataset10']  # x轴刻度的标识

# xspace = np.arange(x_min_log, x_max_log, float((x_max_log - x_min_log)/10.0))
xspace = np.arange(np.min(res[:,0]), np.max(res[:, 0])+0.1, (np.max(res[:, 0]) - np.min(res[:,0])) / 10.0)
yspace = np.arange(y_min, y_max, float((y_max - y_min)/5.0))

group_labels = []
for v in xspace:
    if v >= 3.0:
        group_labels.append('%.d'%(round(v/1000))+'k')
    else:
        group_labels.append('%.d' % (round(v)))

plt.xticks(xspace, group_labels, fontsize=12, fontweight=glb_fontweight)  # 默认字体大小为10
# plt.yticks(yspace, fontsize=12, fontweight=glb_fontweight)
# plt.title("example", fontsize=12, fontweight=glb_fontweight)  # 默认字体大小为12
plt.xlabel("step", fontsize=13, fontweight=glb_fontweight)
plt.ylabel("reward", fontsize=13, fontweight=glb_fontweight)
plt.xlim(x_min, x_max)  # 设置x轴的范围
plt.ylim(y_min, y_max)

# plt.legend()          #显示各曲线的图例
plt.legend(loc=0, numpoints=1)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=12, fontweight=glb_fontweight)  # 设置图例字体的大小和粗细
plt.tight_layout()

plt.savefig(figfile + '.pdf', format='pdf')  # 建议保存为svg格式，再用inkscape转为矢量图emf后插入word中
plt.savefig(figfile + '.png', format='png')  # 建议保存为svg格式，再用inkscape转为矢量图emf后插入word中

if len(sys.argv) < 2:
    plt.show()
print('file is saved to :' + figfile)
