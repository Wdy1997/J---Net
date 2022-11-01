# 导入所需的库
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# 正常显示中文标签
mpl.rcParams['font.sans-serif'] = ['SimHei']

# 自动适应布局
mpl.rcParams.update({'figure.autolayout': True})

# 正常显示负号
mpl.rcParams['axes.unicode_minus'] = False

# 定义颜色，主色：蓝色，辅助色：灰色，互补色：橙色
c = {'蓝色': '#00589F', '深蓝色': '#003867', '浅蓝色': '#5D9BCF',
     '灰色': '#999999', '深灰色': '#666666', '浅灰色': '#CCCCCC',
     '橙色': '#F68F00', '深橙色': '#A05D00', '浅橙色': '#FBC171'}

# 数据源路径
filepath = r"C:\Users\Ad'min\DeepLearning\torch/a.xlsx"

# 读取 Excel文件
df = pd.read_excel(filepath, 1)

# 定义画图用的数据
category_names = df.index
labels = df.columns
data = df.values

df['变化'] = df.iloc[:, 2] - df.iloc[:, 1]

# 使用「面向对象」的方法画图，定义图片的大小
fig, ax = plt.subplots(figsize=(40, 30))
#内部网格线
plt.grid(color='black', linestyle='--', linewidth=1,alpha=0.3)

# 设置背景颜色
fig.set_facecolor('w')
ax.set_facecolor('w')

# 设置标题
plt.title('\nMean_Iou\n\n', loc='center', size=26, color=c['深灰色'])

# 定义范围
rng = range(1, len(df.index) + 1)
rng_pos = list(map(lambda x: x + 1, df[df['变化'] >= 0].index))
rng_neg = list(map(lambda x: x + 1, df[df['变化'] < 0].index))

# 绘制哑铃图中间的线条
# ax.vlines(x=rng_pos, ymin=df[df['变化'] >= 0].iloc[:, 1], ymax=df[df['变化'] >= 0].iloc[:, 2], color=c['浅橙色'], zorder=1,
#           lw=10, label='Up')
# ax.vlines(x=rng_neg, ymin=df[df['变化'] < 0].iloc[:, 1], ymax=df[df['变化'] < 0].iloc[:, 2], color=c['浅蓝色'], zorder=1, lw=10,
#           label='Down')

ax.vlines(x=rng_pos, ymin=df[df['变化'] >= 0].iloc[:, 1], ymax=df[df['变化'] >= 0].iloc[:, 2], color=c['浅橙色'], zorder=1,
          lw=20, label='Up')
ax.vlines(x=rng_neg, ymin=df[df['变化'] < 0].iloc[:, 1], ymax=df[df['变化'] < 0].iloc[:, 2], color=c['浅蓝色'], zorder=1, lw=20,
          label='Down')


# 绘制哑铃图两头的圆点
# ax.scatter(rng, df.iloc[:, 1], color=c['蓝色'], label=df.columns[1], s=80, zorder=2)
# ax.scatter(rng, df.iloc[:, 2], color=c['橙色'], label=df.columns[2], s=80, zorder=2)

ax.scatter(rng, df.iloc[:, 1], color=c['蓝色'], label=df.columns[1], s=400, zorder=2)
ax.scatter(rng, df.iloc[:, 2], color=c['橙色'], label=df.columns[2], s=400, zorder=2)


# 显示数据标签
for i, (txt1, txt2, change) in enumerate(zip(df.iloc[:, 1], df.iloc[:, 2], df.iloc[:, 3])):
    color = c['橙色'] if (change > 0) else c['蓝色']
    radio = (float(txt2) - float(txt1)) / float(txt2)
    label = '+{:.2%}'.format(radio) if radio > 0 else '{:.2%}'.format(radio)
    print(df.iloc[i, 1:3].mean())
    # ax.annotate(label, (df.index[i] + 1.2, df.iloc[i, 2]+2), color=color, ha='center', va='center', fontsize=10)
    ax.annotate(label, (df.index[i] + 1.2, df.iloc[i, 2]+2), color=color, ha='center', va='center', fontsize=20)

# 设置 Y 轴标签
# plt.xticks(rng, df.iloc[:, 0], va='bottom', color=c['深灰色'], size=12)
plt.xticks(rng, df.iloc[:, 0], va='bottom', color=c['深灰色'], size=20)
plt.yticks()
plt.gca().set(
    xlim=(10, 20),
    ylim=(0, 100),
)

# 隐藏边框
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

ymin = df.iloc[:, 1:3].min().min()
ymax = df.iloc[:, 1:3].max().max()
ax.set_ylim(80,  95)
plt.yticks(size=20)
ax.legend()

ax.tick_params(axis='x', which='major', length=0)
plt.tight_layout()
plt.show()