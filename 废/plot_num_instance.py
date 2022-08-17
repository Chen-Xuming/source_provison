import numpy as np
import matplotlib.pyplot as plt

fontsize = 18
linewidth = 2
markersize = 8
plt.rcParams.update({'font.size':fontsize, 'lines.linewidth':linewidth, 'lines.markersize':markersize, 'pdf.fonttype':42, 'ps.fonttype':42})
fontsize_legend = 18
color_list = ['#FF1F5B', '#009ADE',  '#F28522', '#58B272', '#AF58BA', '#A6761D','#1f77b4','#ff7f0e'] 
marker_list = ['o', '^', 'X', 'd', 's', 'v', 'P',  '*','>','<','x']
label_list = ['greedy','greedy-delay-reduction','random']

def plot(x, y):
    fig, ax = plt.subplots()
    max_y = 0
    for i in range(len(y)):
        plt.plot(x, y[i], label=label_list[i], marker=marker_list[i], color=color_list[i])    
        if max(y[i]) > max_y:
            max_y = max(y[i])
    
    plt.legend(fontsize=fontsize_legend)    
    plt.xlabel('Num of instance')
    plt.ylabel('Max interaction delay')
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.ylim([0, max_y])
    plt.xticks(ticks=x)
    filename = 'num instance-max interaction delay0.pdf'
    plt.savefig(filename, transparent=True, bbox_inches='tight', pad_inches=0.04)            
    plt.show()


#result = {'min_max_greedy': [89.23, 115.67, 153.7, 141.11, 194.76, 174.72, 165.69, 261.27, 226.21, 208.27], 'min_max_delay_only': [89.4, 125.88, 151.75, 144.51, 199.78, 187.45, 195.33, 298.24, 216.79, 225.34], 'min_max_random': [127.62, 190.8, 1083.88, 410.2, 374.41, 526.08, 340.6, 912.47, 410.05, 264.65]}
#result ={'min_max_greedy': [93.7, 115.99, 179.34, 157.98, 163.91, 203.25, 261.38, 193.91, 192.05, 303.86], 'min_max_delay_only': [95.22, 122.19, 187.51, 159.12, 163.81, 225.53, 340.61, 196.95, 191.23, 303.86], 'min_max_random': [120.39, 126.1, 430.08, 187.38, 266.2, 294.83, 1250.81, 236.26, 344.97, 453.22]}
#result ={'min_max_greedy': [99.78, 108.54, 121.93, 159.59, 183.07, 218.42, 173.89, 188.61, 205.06, 269.37], 'min_max_delay_only': [99.55, 109.36, 132.51, 167.21, 183.07, 218.42, 170.81, 202.13, 205.06, 268.34], 'min_max_random': [111.72, 224.07, 149.07, 216.74, 486.94, 364.5, 196.52, 256.86, 407.12, 339.28]}
#result ={'min_max_greedy': [108.42, 118.94, 136.38, 143.52, 197.29, 198.8, 181.79, 261.22, 211.16, 183.09], 'min_max_delay_only': [108.42, 117.7, 136.26, 144.86, 201.13, 196.35, 181.79, 248.88, 239.15, 188.78], 'min_max_random': [139.25, 123.96, 181.55, 198.71, 216.34, 1116.59, 267.28, 353.91, 256.22, 244.65]}
#result ={'min_max_greedy': [93.83, 110.88, 157.35, 154.6, 164.62, 170.52, 209.67, 200.57, 214.18, 387.74], 'min_max_delay_only': [96.1, 116.04, 165.94, 157.96, 167.03, 180.84, 209.67, 217.21, 214.18, 372.03], 'min_max_random': [108.03, 125.33, 305.15, 182.43, 334.88, 225.82, 283.99, 311.47, 279.76, 2135.19]}
x = np.arange(0,10)
y = []
y.append(result['min_max_greedy'])
y.append(result['min_max_delay_only'])
y.append(result['min_max_random'])

print(x)
print(y)

plot(x,y)