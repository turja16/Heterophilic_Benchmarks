
from matplotlib import pyplot as plt
import numpy as np
# plots are saved at ./plots
mode = 'RG'
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







