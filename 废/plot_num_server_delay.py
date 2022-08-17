import os
import numpy as np
import matplotlib.pyplot as plt

fontsize = 18
linewidth = 2
markersize = 8
plt.rcParams.update({'font.size':fontsize, 'lines.linewidth':linewidth, 'lines.markersize':markersize, 'pdf.fonttype':42, 'ps.fonttype':42})
fontsize_legend = 18
color_list = ['#FF1F5B', '#009ADE',  '#F28522', '#58B272', '#AF58BA', '#A6761D','#1f77b4','#ff7f0e'] 
marker_list = ['o', '^', 'X', 'd', 's', 'v', 'P',  '*','>','<','x']
label_list = ['λ=100, μ=1', 'λ=100, μ=5', 'λ=100, μ=10','λ=100, μ=50','λ=100, μ=100']

def plot_delay(x, y):
    fig, ax = plt.subplots()
    max_y = 0
    min_y = 1000000

    for i in range(len(y)):
        plt.plot(x, y[i], label=label_list[i], marker=marker_list[i], color=color_list[i])   
        if max(y[i]) > max_y:
            max_y = max(y[i])
        if min(y[i]) < min_y:
            min_y = min(y[i])   
    
    plt.legend(fontsize=fontsize_legend)    
    plt.xlabel('Num server')
    plt.ylabel('Queuing delay(ms)')
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.ylim([0, max_y])
    plt.xticks(ticks=x)

    filename = 'Num server Queuing delay'+'.pdf'
    plt.savefig(filename, transparent=True, bbox_inches='tight', pad_inches=0.04)            
    plt.show() 


def compute_queuing_delay_iteratively(lam, mu, num_server):
    c=0
    while c*mu < lam:
        c+=1
    c += num_server
    r = lam/mu
    rho = r/c
    assert rho < 1    
    
    p0c_2 = 1.
    n = 1
    p0c_1 = r/n
    n += 1
    while n <= c:
        p0c_2 += p0c_1            
        p0c_1 *= r/n
        n += 1

    p0 = 1/(p0c_1/(1-rho) + p0c_2)
    wq = p0c_1*p0/c/(mu * (1-rho)**2)
    assert wq >= 0.
    return wq    


#['λ=100, μ=1', 'λ=100, μ=5', 'λ=100, μ=10','λ=100, μ=50','λ=100, μ=100']

x = [i for  i in range(1,20)]
y = [[compute_queuing_delay_iteratively(100,1,i)*1000 for i in x],
     [compute_queuing_delay_iteratively(100,5,i)*1000 for i in x],
     [compute_queuing_delay_iteratively(100,10,i)*1000 for i in x],
     [compute_queuing_delay_iteratively(100,50,i)*1000 for i in x],
     [compute_queuing_delay_iteratively(100,100,i)*1000 for i in x],
     ]

for i in range(len(y)):
    for j in range(len(y[i])):
        print("{:.2}".format(y[i][j]), ' , ',end="")
        
    print()
plot_delay(x, y)
    