import os
import json
import numpy as np
import matplotlib.pyplot as plt

fontsize = 18
linewidth = 2
markersize = 8
plt.rcParams.update({'font.size':fontsize, 'lines.linewidth':linewidth, 'lines.markersize':markersize, 'pdf.fonttype':42, 'ps.fonttype':42})
fontsize_legend = 18
color_list = ['#FF1F5B', '#009ADE',  '#F28522', '#58B272', '#AF58BA', '#A6761D','#1f77b4','#ff7f0e'] 
marker_list = ['o', '^', 'X', 'd', 's', 'v', 'P',  '*','>','<','x']
label_list = ['min_max_pulp', 'min_max_greedy']#['greedy', 'upper bound', 'lower bound', 'equal weight']


 
def plot_max_delay(min_max_pulp, min_max_greedy):
    print('max_delay_summary')
    x = np.arange(10,110,10)

    fig, ax = plt.subplots()
    max_y = -1
    min_y = 1000000
    for i in range(len(min_max_pulp)):
        plt.plot(x, min_max_pulp[i], label='min_max_pulp user num: ' + str((i+1)*100), marker=marker_list[i], color=color_list[i], linestyle = '-') 
        plt.plot(x, min_max_greedy[i], label='min_max_greedy user num: ' + str((i+1)*100), marker=marker_list[i], color=color_list[i], linestyle = '--')   
        if max(min_max_pulp[i]) > max_y:
            max_y = max(min_max_pulp[i])
        if min(min_max_pulp[i]) < min_y:
            min_y = min(min_max_pulp[i])
        if max(min_max_greedy[i]) > max_y:
            max_y = max(min_max_greedy[i])
        if min(min_max_greedy[i]) < min_y:
            min_y = min(min_max_greedy[i])        
    
    plt.legend(fontsize=fontsize_legend)    
    plt.xlabel('Budget')
    plt.ylabel('Max delay(ms)')
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.ylim([min_y-1, max_y+1])
    plt.xticks(ticks=x)
    plt.yticks(np.arange(int(min_y/10)*10,int(max_y),10))
    filename = 'budget_max delay-100-500user'+'.pdf'
    plt.savefig(filename, transparent=True, bbox_inches='tight', pad_inches=0.04)            
    plt.show() 


def plot_approximate_rate_to_optimal(rate):
    width = 1
    offset_w = 0.5
    n = len(min_max_pulp)
    if n % 2 == 0:
        offset = np.arange(1-n, n+1, 2) * width * offset_w
    else:            
        offset = np.arange(-n+1, n+1, 2) * width * offset_w    
    
    
    x = np.arange(10,110,10)
    fig, ax = plt.subplots()
    

    max_rate = -1
    for i in range(len(rate)):
        plt.bar(x+offset[i], rate[i], color=color_list[i], label='min_max_greedy user num: ' + str((i+1)*100), width = width)
        if max_rate<max(rate[i]):
            max_rate = max(rate[i])

    
    plt.legend(fontsize=fontsize_legend)    
    plt.xlabel('Budget')
    plt.ylabel('Approximate rate to optimal')
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.ylim([1, max_rate+0.01])
    plt.xticks(ticks=x)
    filename = 'Approximate rate to optimal'+'.pdf'
    plt.savefig(filename, transparent=True, bbox_inches='tight', pad_inches=0.04)            
    plt.show()     
 



algorithm_name_list = ['min_max_pulp', 'min_max_greedy']#[ 'min_max_greedy', 'upper bound','lower bound', 'min_max_equal_weight']

min_max_pulp =[  [203.70091647402197, 164.8609500859078, 117.14733260923498, 102.69502234886536, 45.638574368637116, 35.52822047283095, 29.113304552260992, 24.420928131395193, 21.2616103373939, 18.343218729466113],           
                 [103.43391044344298, 82.28167550285428, 71.54334682891061, 64.25865954483092, 58.017457942798146, 48.178416628413885, 24.167984891465995, 20.771513518455766, 18.77298021166824, 17.2025666177793],
                 [113.43033388309506, 98.20052034834265, 63.053657994156076, 57.835355071194236, 50.758289111394, 47.60323309744479, 45.23932898680623, 41.494771145843316, 39.42959636317361, 37.29045659245515],
                 [65.10914699566422, 50.340064543452485, 43.423792194221114, 38.91977381301334, 35.79500277453361, 33.62374273507153, 31.465985793688716, 28.21643793958926, 23.82870425684364, 20.34858124813082],                 
                 [61.51708295278135, 46.37638875567727, 40.37914788055925, 37.60578068768035, 35.15872145793174, 33.477751239872276, 31.739217100335104, 30.538465271543263, 29.326473247799207, 28.30687599310289]                 
                 ]



min_max_greedy =[  [205.6803802453122, 167.12845436792645, 120.36181719799808, 105.5721198113272, 51.07055199649172, 41.03255403689787, 36.754440406999784, 27.13809253002453, 23.417484267005193, 21.080783794057506],
                   [105.25560029101165, 83.38032360032022, 73.41937729869562, 66.97803524165748, 63.968278351059936, 56.880198559477606, 46.30670152417351, 22.86064627698599, 20.825701370364456, 19.249891359494487],                   
                   [114.34600203014848, 99.43362913427272, 64.32975178084935, 60.390162769267434, 57.02868737816441, 49.73985539191787, 46.94181695056273, 45.45595002550015, 41.839234804685766, 40.80127840959905],
                   [66.12149625626583, 51.23133111848671, 46.716257812984075, 40.72970986536727, 38.674891285418354, 35.59531217467191, 33.21212275839425, 31.219149215066995, 30.375651370359947, 22.101467862068684],                   
                   [63.947804673370676, 47.67428647533122, 41.19042639311314, 38.387630005142704, 36.502570380975456, 34.48110941742169, 32.77967827341077, 31.52014040582175, 30.721906738941033, 29.896521795746008]
                   ]


rate=[
    [1.02758107112475, 1.0358815718568342, 1.0471321662684618, 1.0816100219641331, 1.1229017527176672, 1.114717224905342, 1.1632791658792607, 1.1118332138023974, 1.1205861409863527, 1.169628496702278],
    [1.0339160628073436, 1.0252278913718678, 1.0415524253345114, 1.0641174809830836, 1.1633687561312644, 1.1666973617656138, 1.453572163926259, 1.1045281100389368, 1.1171918714403666, 1.1299201824076075],
    [1.0126989356378615, 1.0302554698141042, 1.031322722123787, 1.0647484601219355, 1.124614388362059, 1.0909728697660797, 1.0661572424324774, 1.162699072700124, 1.1261224799728515, 1.1760681937222637],
    [1.018691987386815, 1.0253969240231142, 1.0776595105209867, 1.055721494368755, 1.1089946814197247, 1.0891492660988826, 1.0678326107869944, 1.0698951196384776, 1.1450487937584597, 1.0881187725489823],
    [1.0373424617054003, 1.0379106939849507, 1.0257053461869952, 1.0301312472378767, 1.0515262344503475, 1.0538407047287095, 1.0546574656352399, 1.0522604084643037, 1.0753681951113307, 1.1031501500284155],    
    ]

plot_max_delay(min_max_pulp, min_max_greedy)
plot_approximate_rate_to_optimal(rate)