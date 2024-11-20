import torch
from matplotlib import pyplot as plt
import numpy as np
# plots are saved at ./plots
# for PA adn Gencat
mode = 'PA' # or Gencat
res = torch.load(f'./{mode}.pt') 
metrics = list(res.keys())  
for metric in metrics:
    all_level_res = res[metric]
    levels = np.array(list(all_level_res.keys()))
    mean = []
    std = []
    for level in levels:
        mean.append(np.mean(all_level_res[level]))
        std.append(np.std(all_level_res[level]))
    #
    mean = np.array(mean)
    std = np.array(std)
    # Plotting
    upper = mean + std
    lower = mean - std
    plt.figure(figsize=(8, 8))
    plt.plot(levels, mean, label=metric, color="blue")  # Plot mean line
    plt.fill_between(levels, lower, upper, color="blue", alpha=0.2)  # Plot shaded area
    plt.xlabel("Homophily Coefficients")
    plt.ylabel("Homophily Values")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./plots/{mode}_{metric}.png', dpi=300)  # Save with high resolution


mode = 'RG'
res = torch.load(f'./{mode}.pt') 
metrics = list(res.keys())
# get the x-axis
all_level_res = res['Edge Homophily']
levels = np.array(list(all_level_res.keys()))
mean = []
for level in levels:
    mean.append(np.mean(all_level_res[level]))
#
mean = np.array(mean) 
x = mean

for metric in metrics:
    if metric == 'Edge Homophily':
        pass 
    all_level_res = res[metric]
    levels = np.array(list(all_level_res.keys()))
    mean = []
    std = []
    for level in levels:
        mean.append(np.mean(all_level_res[level]))
        std.append(np.std(all_level_res[level]))
    #
    mean = np.array(mean)
    std = np.array(std)
    # Plotting
    upper = mean + std
    lower = mean - std
    plt.figure(figsize=(8, 8))
    plt.plot(x, mean, label=metric, color="blue")  # Plot mean line
    plt.fill_between(x, lower, upper, color="blue", alpha=0.2)  # Plot shaded area
    plt.xlabel("Computed Edge Homophily")
    plt.ylabel("Homophily Values")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./plots/{mode}_{metric}.png', dpi=300)  # Save with high resolution




