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


 
def plot_max_delay(x, min_max_pulp, min_max_greedy):
    print('max_delay_summary')
    

    fig, ax = plt.subplots()
    max_y = -1
    min_y = 1000000
    for i in range(len(min_max_pulp)):
        plt.plot(x, min_max_pulp[i], label='min_max_pulp service num: ' + str((i+1)*5), marker=marker_list[i], color=color_list[i], linestyle = '-') 
        plt.plot(x, min_max_greedy[i], label='min_max_greedy service num: ' + str((i+1)*5), marker=marker_list[i], color=color_list[i], linestyle = '--')   
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


def plot_approximate_rate_to_optimal(x, rate):
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
        plt.bar(x+offset[i], rate[i], color=color_list[i], label='min_max_greedy service num: ' + str((i+1)*5), width = width)
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

min_max_pulp =[  [212.0457246252305, 84.33210818380006, 71.19038666641949, 57.761794482866826, 52.311602315669724, 28.496327101662477, 22.211701435074318, 19.13833966219722, 16.45964109398222, 13.426351468471546],
                  [203.70091647402197, 164.8609500859078, 117.14733260923498, 102.69502234886536, 45.638574368637116, 35.52822047283095, 29.113304552260992, 24.420928131395193, 21.2616103373939, 18.343218729466113],
                  [258.60021333542943, 103.43350901495549, 71.1458076346126, 51.1002151294188, 43.199748807269856, 35.62277068392832, 31.390759678984242, 28.153026299682278, 25.28499881139915, 22.870544980783723],    
                  [230.18112553729603, 131.06500757852956, 89.22650813284196, 60.656159319976176, 48.58080887369519, 40.72782809950249, 32.82613448112221, 28.852590845597543, 25.707707582831485, 23.015534503426643],                  
                  [693.0425770532429, 228.45773987270206, 103.32785184487392, 62.902713283849806, 49.2329158008583, 40.933611629819126, 33.99625794626943, 29.120764128853466, 25.864972879812772, 23.388377504113983]
                
                  ]



min_max_greedy =[  [215.0398684647011, 85.01750940423739, 79.14425965325812, 60.02321429394477, 53.517778419840084, 40.89549862842236, 23.33625054151262, 21.289137731314327, 18.27439636763031, 15.797737328315723],
                    [205.6803802453122, 167.12845436792645, 120.36181719799808, 105.5721198113272, 51.07055199649172, 41.03255403689787, 36.754440406999784, 27.13809253002453, 23.417484267005193, 21.080783794057506],
                    [260.34141667808717, 107.03757647440416, 73.72488659583568, 55.185009771000665, 47.72262638335377, 38.16194264934816, 33.78816672271008, 31.156484031537182, 28.53080172472572, 25.638578115562527],       
                     [234.5317520451613, 138.31278928711222, 92.83496509245684, 63.57746804634978, 51.22671121615219, 45.4733158622633, 36.58110439413949, 31.81453127273906, 28.276399669073797, 26.10613021309989],       
                    [693.7117659779135, 232.04098085426673, 106.07411003730587, 65.29990905474014, 52.123773192274136, 43.11602489307856, 36.88195981871081, 31.247001603546387, 28.65044911660807, 25.795339575198778]

                    
                    ]
rate = [
    [1.0461581065845211, 1.0163459767598515, 1.0878868105677413, 1.0605816717698406, 1.0736623305451671, 1.208387232511287, 1.0962337055227582, 1.1977114215499904, 1.1490574117852712, 1.20516898021536],
    [1.02758107112475, 1.0358815718568342, 1.0471321662684618, 1.0816100219641331, 1.1229017527176672, 1.114717224905342, 1.1632791658792607, 1.1118332138023974, 1.1205861409863527, 1.169628496702278],    
    [1.0170286325942421, 1.048742876331432, 1.046434358573592, 1.0755564991387236, 1.0955666327981652, 1.0751107388049275, 1.0711054267041566, 1.107329273175831, 1.1429744812058582, 1.1374768150134325],
    [1.0314777675459588, 1.0673346534154808, 1.0525980628715954, 1.0598042842452498, 1.0547081779013452, 1.0969200685598228, 1.102895326100784, 1.1032413765570985, 1.1074604662898917, 1.1275195422676334],
    [1.0027365009264948, 1.0411711279134426, 1.0532118412350688, 1.0477814644460473, 1.0721988116113448, 1.0616693943272948, 1.0854351744548032, 1.079352569108254, 1.1112301430106855, 1.1093936304471732],    
]


x = np.arange(10,110,10)
plot_max_delay(x, min_max_pulp, min_max_greedy)
plot_approximate_rate_to_optimal(x, rate)