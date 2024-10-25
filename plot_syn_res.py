from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

linecolor = ['#b07aa1', '#ff9da7', '#59a14f', '#76b7b2']
facecolor = ['#c8aedf', '#ffc2cb', '#a1dbb2', '#a0d4cf']

use_mixhop = True

if use_mixhop:
    h = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
else:
    h = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]

model = ['mlp1', 'mlp2', 'gcn', 'sgc1']
label = ['MLP-1 Performance', 'MLP-2 Performance', 'GCN Performance', 'SGC Performance']
BASE_DIR = "./res_search_syn"
name = 'mixhop_summary.csv' if use_mixhop else 'gencat_summary.csv'
df = pd.read_csv('{}/{}'.format(BASE_DIR, name), header=None)

column_names = ['model', 'h', 'mean', 'std'] if use_mixhop else ['model', 'base', 'h', 'mean', 'std']
df.columns = column_names
plt.clf()
l=0.8
fig, ax = plt.subplots(figsize=(5.5, 4.0))
for i in range(len(model)):
    view_model = df[df['model'].str.strip() == model[i]]
    view_model = view_model.sort_values(by=['h'])
    plt.plot(h, view_model['mean'].values, color=linecolor[i], label=f'{label[i]}', linewidth=l)
    plt.fill_between(h,  
        view_model['mean'].values - view_model['std'].values, 
        view_model['mean'].values + view_model['std'].values,
        alpha=0.1, facecolor=facecolor[i],
        linewidth=2, linestyle='dashdot', antialiased=True)

text = r'$\mu$' if use_mixhop else r'$\beta$'
ax.set_xlabel(text, fontsize=12)
# ax.set_ylabel("Accuracy", fontsize=12)
# ax.set_title('Preferential Attachment (sym)')
if use_mixhop:
    x_ticks = np.arange(0.0, 1.0, 0.1)
else:
    x_ticks = np.arange(-1.0, 9.0, 1.0)

plt.xticks(x_ticks)
if not use_mixhop:
    print('set legend')
    # leg = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    leg = ax.legend(loc='upper right')

fig.tight_layout()
name = 'fig_pa_baseline.eps' if use_mixhop else 'fig_gencat_baseline.eps'
plt.savefig(name)  # Save the figure

   


        

