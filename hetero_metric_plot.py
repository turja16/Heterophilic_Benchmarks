from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch 

use_mixhop = False

if use_mixhop:
    # mixhop
    h = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
else:
    # cora
    h = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]

H_node = []
H_edge = []
H_class = []
H_ge = []
H_m_AGG = []
H_edge_adj = []
LI = []
H_nei = []
kernel_reg0 = []
kernel_reg1 = []
gnb = []
color = ['#b07aa1', '#ff9da7', '#e15759', '#f28e2b', '#edc948', '#59a14f', '#76b7b2', '#4e79a7', '#9c755f', '#bab0ac', '#000000']
for i in range(len(h)):
    hi = h[i]
    for idx in range(10):
        if use_mixhop:
            name = './stat/mixhop_h{}_g{}'.format(hi, idx)
        else:
            name = './stat/gencat_{}_{}_{}'.format('cora', hi, idx)
        res = torch.load('{}.pt'.format(name))
        H_node.append(res['H node'])
        H_edge.append(res['H edge'])
        H_class.append(res['H class'])
        H_ge.append(res['H_ge'])
        H_m_AGG.append(res['H_m_AGG_sym'])
        H_edge_adj.append(res['H_edge_adj'])
        LI.append(res['LI'])
        H_nei.append(res['H_nei'])
        kernel_reg0.append(res['kernel_reg0_sym'])
        kernel_reg1.append(res['kernel_reg1_sym'])
        gnb.append(res['gnb_sym'])

def take_mean(x, size):
    x_mean = []
    i = 0
    # take mean [i, i+size)
    while i < len(x):
        x_mean.append(sum(x[i:i+size])/size)
        i = i + size
    return x_mean

# take average over 10 graphs for each level
H_node = np.array(take_mean(H_node, 10))
H_edge = np.array(take_mean(H_edge, 10))
H_class = np.array(take_mean(H_class, 10))
H_ge = np.array(take_mean(H_ge, 10))
H_m_AGG = np.array(take_mean(H_m_AGG, 10))
H_edge_adj = np.array(take_mean(H_edge_adj, 10)) 
LI = np.array(take_mean(LI, 10))
H_nei = np.array(take_mean(H_nei, 10))
kernel_reg0 = np.array(take_mean(kernel_reg0, 10)) 
kernel_reg1 = np.array(take_mean(kernel_reg1, 10))
gnb = np.array(take_mean(gnb, 10))

if use_mixhop:
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
else:
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

#plt.rcParams.update(plt.rcParamsDefault)
l=0.8
plt.plot(h, H_node, color='orange', label=r'$H_{\mathrm{node}}(\mathcal{G})$', linewidth=l)
plt.plot(h, H_edge, color='green', label=r'$H_{\mathrm{edge}}(\mathcal{G})$', linewidth=l)
plt.plot(h, H_class, color='black', label=r'$H_{\mathrm{class}}(\mathcal{G})$', linewidth=l)
plt.plot(h, H_ge, color='yellow', label=r'$H_{\mathrm{GE}}(\mathcal{G})$', linewidth=l)
plt.plot(h, H_m_AGG, color='purple', label=r'$H_{\mathrm{agg}}(\mathcal{G})$', linewidth=l)
plt.plot(h, H_edge_adj, color='blue', label=r'$H_{\mathrm{adj}}(\mathcal{G})$', linewidth=l)
plt.plot(h, LI, color='grey', label=r'LI', linewidth=l)
plt.plot(h, 1-H_nei, color='red', label=r'$1-H_{\mathrm{neighbor}}(\mathcal{G})$', linewidth=l)
plt.plot(h, kernel_reg0, 'teal', label=r'$KR_{L}$', linewidth=l)
plt.plot(h, kernel_reg1, color='salmon', label=r'$KR_{NL}$', linewidth=l)
plt.plot(h, gnb, color='peru', label=r'GNB', linewidth=l)
text = r'$\mu$' if use_mixhop else r'$\beta$'
ax.set_xlabel(text, fontsize=12)
# ax.set_ylabel("Metrics", fontsize=12)
if use_mixhop:
    x_ticks = np.arange(0.0, 1.0, 0.1)
else:
    x_ticks = np.arange(-1, 14, 1)

if not use_mixhop:
    # leg = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    leg = ax.legend(loc='upper right')


plt.xticks(x_ticks)
# ax.set_title('Preferential Attachment (sym)')
# ax.set_title('GenCat (rw)')
fig.tight_layout()
# plt.savefig("mixhop_heter_metric_sym.png") 
name = "fig_pa_metrics.eps" if use_mixhop else "fig_gencat_metrics.eps"
plt.savefig(name) 




