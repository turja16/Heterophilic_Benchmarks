import torch
from collections import defaultdict
# metric = ['edge', 'node', 'class', 'li', 'adjust', 'ge', 'agg', 'ne', 'kernel_reg0', 'kernel_reg1', 'gnb']
results = defaultdict(dict)
# mode = 'PA'
# levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# mode = 'Gencat'
# levels = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
mode = 'RG'
levels = [0.05, 0.1, 0.15, 0.16, 0.165, 0.17, 0.175, 0.18, 0.185, 0.19, 0.195, 0.2, 0.21, 0.22, 0.23, 0.24,
                    0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

for level in levels:
    res = torch.load(f'./{mode}/edge_{level}.pt')
    results['Edge Homophily'][level] = res
    #
    res = torch.load(f'./{mode}/node_{level}.pt')
    results['Node Homophily'][level] = res
    #
    res = torch.load(f'./{mode}/class_{level}.pt')
    results['Class Homophily'][level] = res
    # 
    res = torch.load(f'./{mode}/li_{level}.pt')
    results['Label Informativeness'][level] = res 
    #
    res = torch.load(f'./{mode}/adjust_{level}.pt')
    results['Adjusted Homophily'][level] = res 
    #
    res = torch.load(f'./{mode}/ge_{level}.pt')
    results['Generalized Edge Homophily'][level] = res    
    #
    res = torch.load(f'./{mode}/agg_{level}.pt')
    results['Aggregation Homophily'][level] = res  
    #
    res = torch.load(f'./{mode}/ne_{level}.pt')
    results['Neighborhood Identifiability'][level] = res 
    #
    res = torch.load(f'./{mode}/kernel_reg0_{level}.pt')
    results['kernel_reg0'][level] = res
    #
    res = torch.load(f'./{mode}/kernel_reg1_{level}.pt')
    results['kernel_reg1'][level] = res
    #
    res = torch.load(f'./{mode}/gnb_{level}.pt')
    results['gnb'][level] = res

torch.save(results, f'./{mode}.pt')
