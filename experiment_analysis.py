import os
import numpy as np

from pymoo.model.evaluator import Evaluator
from pymoo.factory import get_problem, get_reference_directions
from pymoo.factory import get_performance_indicator
from pymoo.model.population import Population, pop_from_array_or_individual

problem = 'DTLZ2'
original_dimension = 5
reduced_dimension = 2
interval_of_aggregations = 1
save_data = True
use_normalization=True
use_different_seeds = True
termination_criterion = ('n_gen', 10)
benchmark_problem = get_problem(problem.lower(), n_obj=original_dimension)
number_of_executions = 5
reference_directions = get_reference_directions("das-dennis", original_dimension, n_partitions=12)

algorithm_1 = 'NSGA3_{}_{}'.format(problem, original_dimension)
results_path_1 = '.\\experiment_results\\'
algorithm_2 = 'OnlineClusterNSGA3_{}_{}_{}_{}'.format(problem, original_dimension, reduced_dimension, interval_of_aggregations)
results_path_2 = '.\\experiment_results\\'
algorithm_3 = 'OfflineClusterNSGA3_{}_{}_{}'.format(problem, original_dimension, reduced_dimension)
results_path_3 = '.\\experiment_results\\'
algorithm_4 = 'RandomClusterNSGA3_{}_{}_{}_{}'.format(problem, original_dimension, reduced_dimension, interval_of_aggregations)
results_path_4 = '.\\experiment_results\\'

def read_variables(variables_path):
    with open(variables_path, "r") as file_data:
        lines = file_data.readlines()
        lines = ' '.join(lines)
        lines = lines.replace('\n','').replace('[','').split(']')
        lines = [line.strip().split() for line in  lines if len(line) > 0]
        lines = [[float(number) for number in line] for line in lines] 
        return np.array(lines)

def get_metrics_for_comparison(number_of_executions, results_path, algorithm, last_file):
    hv_final = []
    igd_final = []
    for i in range(number_of_executions):
        print('Execution {}'.format(i))
        variables_path = os.path.join(results_path, algorithm, 'Execution {}'.format(i), last_file)
        read_pop = pop_from_array_or_individual(read_variables(variables_path))
        evaluator = Evaluator()
        evaluator.eval(benchmark_problem, read_pop)

        hv = get_performance_indicator("hv", ref_point=np.array([1.2]*benchmark_problem.n_obj))
        igd_plus = get_performance_indicator("igd+", benchmark_problem.pareto_front(ref_dirs=reference_directions))

        #hv_final.append(hv.calc(read_pop.get('F')))
        igd_final.append(igd_plus.calc(read_pop.get('F')))
    return hv_final, igd_final

hv_1, igd_1 = get_metrics_for_comparison(number_of_executions, results_path_1, algorithm_1, 'variables_008.txt')
hv_2, igd_2 = get_metrics_for_comparison(number_of_executions, results_path_2, algorithm_2, 'variables_008.txt')
hv_3, igd_3 = get_metrics_for_comparison(number_of_executions, results_path_3, algorithm_3, 'variables_008.txt')
hv_4, igd_4 = get_metrics_for_comparison(number_of_executions, results_path_4, algorithm_4, 'variables_008.txt')