import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_data_file(file_path):
    with open(file_path, 'r') as file:
        return [float(line.replace('\n','')) for line in file.readlines()]

def generate_convergence_plot_full(file_name, save_files_path, use_log, results_path_1, algorithm_1,
 results_path_2, algorithm_2, results_path_3, algorithm_3, results_path_4, algorithm_4):
    if use_log:
        metric_1 = np.log(read_data_file(os.path.join(results_path_1, algorithm_1, file_name)))
        metric_2 = np.log(read_data_file(os.path.join(results_path_2, algorithm_2, file_name)))
        metric_3 = np.log(read_data_file(os.path.join(results_path_3, algorithm_3, file_name)))
        metric_4 = np.log(read_data_file(os.path.join(results_path_4, algorithm_4, file_name)))
    else:
        metric_1 = read_data_file(os.path.join(results_path_1, algorithm_1, file_name))
        metric_2 = read_data_file(os.path.join(results_path_2, algorithm_2, file_name))
        metric_3 = read_data_file(os.path.join(results_path_3, algorithm_3, file_name))
        metric_4 = read_data_file(os.path.join(results_path_4, algorithm_4, file_name))
    plt.figure()
    plt.plot(metric_1, label=algorithm_1)
    plt.plot(metric_2, label=algorithm_2)
    plt.plot(metric_3, label=algorithm_3)
    plt.plot(metric_4, label=algorithm_4)
    plt.xlabel('Generation', fontsize=20)
    plt.ylabel(file_name.split('.')[0], fontsize=20)
    plt.legend()
    plt.savefig(os.path.join(save_files_path, file_name.split('.')[0] + '.pdf'))
    plt.show()

def generate_boxplots_full(number_of_executions, file_name, save_files_path, results_path_1, algorithm_1,
 results_path_2, algorithm_2, results_path_3, algorithm_3, results_path_4, algorithm_4):
    data_1 = []
    data_2 = []
    data_3 = []
    data_4 = []
    for i in range(number_of_executions):
        data_1.append(read_data_file(
            os.path.join(results_path_1, algorithm_1, 'Execution {}'.format(i), file_name))[-1])
        data_2.append(read_data_file(
            os.path.join(results_path_2, algorithm_2, 'Execution {}'.format(i), file_name))[-1])
        data_3.append(read_data_file(
            os.path.join(results_path_3, algorithm_3, 'Execution {}'.format(i), file_name))[-1])
        data_4.append(read_data_file(
            os.path.join(results_path_4, algorithm_4, 'Execution {}'.format(i), file_name))[-1])
    
    plt.title(file_name.split('_')[0])
    plt.boxplot([data_1, data_2, data_3, data_4], labels=[
        algorithm_1, 
        algorithm_2.replace('ClusterNSGA3',''),
        algorithm_3.replace('ClusterNSGA3',''),
        algorithm_4.replace('ClusterNSGA3','')])
    plt.xticks(rotation=10)
    plt.savefig(os.path.join(save_files_path, file_name.split('_')[0] + '_boxplot.pdf'))
    plt.show()

def create_folder(full_path):
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        print('Final results folder created!')
    else:
        print('Folder already exists!')

problem = 'DTLZ2'
original_dimension = 5
reduced_dimension = 4
number_of_executions = 5
interval_of_aggregations = 1

# algorithm files path
algorithm_1 = 'NSGA3_{}_{}'.format(problem, original_dimension)
results_path_1 = '.\\experiment_results\\'
algorithm_2 = 'OnlineClusterNSGA3_{}_{}_{}_{}'.format(problem, original_dimension, reduced_dimension, interval_of_aggregations)
results_path_2 = '.\\experiment_results\\'
algorithm_3 = 'OfflineClusterNSGA3_{}_{}_{}'.format(problem, original_dimension, reduced_dimension)
results_path_3 = '.\\experiment_results\\'
# algorithm_3 = algorithm_1
# results_path_3 = results_path_1
algorithm_4 = 'RandomClusterNSGA3_{}_{}_{}_{}'.format(problem, original_dimension, reduced_dimension, interval_of_aggregations)
results_path_4 = '.\\experiment_results\\'
save_files_path = '.\\experiment_results\\{}_{}_{}_{}'.format(problem, original_dimension, reduced_dimension, interval_of_aggregations)
hv_file_name = 'mean_hv_convergence.txt'
igd_file_name = 'mean_igd_convergence.txt'

create_folder(save_files_path)

## Convergence plots
# generate_convergence_plot_full(hv_file_name, save_files_path, results_path_1, algorithm_1,
#  results_path_2, algorithm_2, results_path_3, algorithm_3, results_path_4, algorithm_4)
generate_convergence_plot_full(igd_file_name, save_files_path, True, results_path_1, algorithm_1,
 results_path_2, algorithm_2, results_path_3, algorithm_3, results_path_4, algorithm_4)

## Boxplots
# generate_boxplots_full(number_of_executions, 'hv_convergence.txt', save_files_path, results_path_1, algorithm_1,
#  results_path_2, algorithm_2, results_path_3, algorithm_3, results_path_4, algorithm_4)
generate_boxplots_full(number_of_executions, 'igd_convergence.txt',save_files_path, results_path_1, algorithm_1,
 results_path_2, algorithm_2, results_path_3, algorithm_3, results_path_4, algorithm_4)