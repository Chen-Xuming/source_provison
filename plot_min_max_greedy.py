import os
import json
import numpy as np
import matplotlib.pyplot as plt
import math

"""
    style of figures
"""
fontsize = 18
linewidth = 2
markersize = 8
plt.rcParams.update({'font.size':fontsize, 'lines.linewidth':linewidth, 'lines.markersize':markersize, 'pdf.fonttype':42, 'ps.fonttype':42})
fontsize_legend = 18
color_list = ['#FF1F5B', '#009ADE',  '#F28522', '#58B272', '#AF58BA', '#A6761D','#1f77b4','#ff7f0e']
marker_list = ['o', '^', 'X', 'd', 's', 'v', 'P',  '*','>','<','x']

# 获取一组实验的json文件的路径
def get_json_file_list(dir_path):
    files = []
    for file_name in os.listdir(dir_path):
        _, postfix = os.path.splitext(file_name)
        if postfix == ".json":
            json_file = os.path.join(dir_path, file_name)
            files.append(json_file)
    return files

"""
    读取json数据
    max_delay = {
        "greedy": {
            10: [d1, d2, ..., d50],
            20: [...],
            ...
        },
        "euqal_weight":...
    }
"""
def read_data(json_file_list, keys, algorithms):
    max_delay = {}
    running_time = {}

    for alg in algorithms:
        max_delay[alg] = {}
        running_time[alg] = {}
        for i in keys:
            max_delay[alg][i] = []
            running_time[alg][i] = []

    for json_file in json_file_list:
        data = json.load(open(json_file))
        for i in keys:
            data_budget = data[str(i)]
            for alg in algorithms:
                max_delay[alg][i].append(data_budget[alg]["max_delay"] * 1000)  # ms
                running_time[alg][i].append(data_budget[alg]["running_time"])
    return max_delay, running_time

def analyse(max_delay, running_time, algorithms, keys):
    result_max_delay = [[] for i in algorithms]
    result_running_time = [[] for i in algorithms]

    for i, alg in enumerate(algorithms):
        for budget in keys:
            result_max_delay[i].append(np.mean(max_delay[alg][budget]))
            result_running_time[i].append(np.mean(running_time[alg][budget]))

    # 近似率: 求各个simulation下的的近似率，然后求平均
    opt_alg = "min_max_pulp"
    approximate_ratio = {}
    for alg in algorithms:
        if alg != opt_alg:
            approximate_ratio[alg] = []
    pulp_index = algorithms.index(opt_alg)
    simulations = len(max_delay[opt_alg][keys[0]])
    for i, alg in enumerate(algorithms):
        if alg != opt_alg:
            for key in keys:
                ratios = []
                for simulation in range(simulations):
                    ratios.append(max_delay[alg][key][simulation] / max_delay[opt_alg][key][simulation])
                approximate_ratio[alg].append(np.mean(ratios))

    # for i, alg in enumerate(algorithms):
    #     if alg != "min_max_pulp":
    #         ratios = []
    #         for j in range(len(keys)):
    #             ratios.append(running_time[1])
    #             approximate_ratio[alg].append(result_max_delay[i][j] / result_max_delay[pulp_index][j])

    return result_max_delay, result_running_time, approximate_ratio

def draw_delay(max_delay, algorithms, keys, save_dir):
    plt.figure()
    plt.xlabel("Budget")
    plt.ylabel("Max Delay (ms)")
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.xticks(ticks=keys)
    min_y = np.min(max_delay)
    max_y = np.max(max_delay)
    plt.yticks(np.arange(int(min_y / 10) * 10, math.ceil(max_y / 10) * 10, 10))
    for i in range(len(algorithms)):
        label = algorithms[i]
        if algorithms[i].find("server_") != -1:
            server_num = int(algorithms[i].split("_")[1])
            label = "min_max_greedy({})".format(server_num)
        plt.plot(keys, max_delay[i], label=label, color=color_list[i], marker=marker_list[i])

    plt.legend(fontsize=fontsize_legend)
    plt.savefig(save_dir + "/max_delay.png", bbox_inches="tight")
    plt.show()

def draw_running_time(running_time, algorithms, keys, save_dir):
    plt.figure()
    plt.xlabel("Budget")
    plt.ylabel("Running Time (ms)")
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.xticks(ticks=keys)
    min_y = np.min(running_time)
    max_y = np.max(running_time)
    # plt.yticks(np.arange(int(min_y / 10) * 10, math.ceil(max_y / 10) * 10, 10))
    plt.ylim([min_y - 100, max_y + 100])
    for i in range(len(algorithms)):
        label = algorithms[i]
        if algorithms[i].find("server_") != -1:
            server_num = int(algorithms[i].split("_")[1])
            label = "min_max_greedy({})".format(server_num)
        plt.plot(keys, running_time[i], label=label, color=color_list[i], marker=marker_list[i])
    plt.legend(fontsize=fontsize_legend)
    plt.savefig(save_dir + "/running_time.png", bbox_inches="tight")
    plt.show()

def draw_approximate_ratio(approximate_ratio, keys, save_dir):
    from matplotlib.patches import Patch

    width = 1
    offset_w = 0.5
    n = len(approximate_ratio)
    if n % 2 == 0:
        offset = np.arange(1 - n, n + 1, 2) * width * offset_w
    else:
        offset = np.arange(-n + 1, n + 1, 2) * width * offset_w

    plt.figure()
    plt.xlabel("Budget")
    plt.ylabel("Approximate Ratio")

    max_rate = 0
    for key, value in approximate_ratio.items():
        max_rate = max(max_rate, max(value))

    plt.tight_layout()
    plt.ylim([1, max_rate + 0.01])
    plt.xticks(ticks=keys)

    for i, alg in enumerate(approximate_ratio):
        for j in range(len(keys)):
            plt.bar(keys[j] + offset[i], approximate_ratio[alg][j], width=width, color=color_list[i])

    patches = [Patch(facecolor=color_list[i], label= alg if alg.find("server_") == -1 else "min_mac_greedy({})".format(alg.split("_")[-1])) for i, alg in enumerate(approximate_ratio)]

    plt.legend(fontsize=fontsize_legend, handles=patches)
    plt.grid(linestyle='--')
    plt.savefig(save_dir + "/approximate_ratio.png", bbox_inches="tight")
    plt.show()


def draw():
    users = 400

    dir_path = "result/2022-7-15-remove-redundant/{}u-10s".format(users)
    keys = [i for i in range(10, 110, 10)]
    algorithms = ['min_max_pulp']

    file_list = get_json_file_list(dir_path)
    max_delay, running_time = read_data(file_list, keys, algorithms)

    dir_path = "result/min_max_greedy_different_server_each_allocation/{}u-10s".format(users)
    algorithm_greedy = ["server_{}".format(i) for i in range(1, 6)]

    file_list = get_json_file_list(dir_path)
    max_delay_ew, runnint_time_ew = read_data(file_list, keys, algorithm_greedy)

    max_delay.update(max_delay_ew)
    running_time.update(runnint_time_ew)
    algorithms += algorithm_greedy

    result_max_delay, result_runnint_time, approximate_ratio = analyse(max_delay, running_time, algorithms, keys)
    # print(result_max_delay)
    # print(result_runnint_time)
    # print(approximate_ratio)

    save_dir = dir_path + "/figure"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    draw_delay(result_max_delay, algorithms, keys, save_dir)
    draw_running_time(result_runnint_time, algorithms, keys, save_dir)
    draw_approximate_ratio(approximate_ratio, keys, save_dir)

if __name__ == "__main__":
    draw()