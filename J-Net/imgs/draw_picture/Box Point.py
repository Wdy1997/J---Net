import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')
plt.rc("font",family="SimHei",size="15")  #解决中文乱码问题

filepath = r"C:\Users\Ad'min\Desktop\点集\M_J_RA_bd_loss\point.xlsx"
df = pd.read_excel(filepath,0)

plt.figure(figsize=(13,10), dpi= 80)


sns.boxplot(x='Method', y='Dice', data=df)

sns.stripplot(x='Method', y='Dice', data=df, color='black', size=3, jitter=1)

for i in range(len(df['Method'].unique())-1):

    plt.vlines(i+.5, 0, 1, linestyles='solid', colors='gray', alpha=0.2)

# Decoration

plt.title('Box Plot of The train set Kvasir&CVC-ClinicDB test set Kvasir', fontsize=20)


plt.show()
