import time
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from pymoo.algorithms.experiment_nsga3 import ExperimentNSGA3
from pymoo.algorithms.adapted_nsga3 import NSGA3
from pymoo.algorithms.online_cluster_nsga3 import OnlineClusterNSGA3
from pymoo.algorithms.offline_cluster_moead import OfflineClusterMOEAD, AggregatedGeneticAlgorithm
from pymoo.factory import get_problem, get_reference_directions
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.experiment_online_cluster_nsga3 import ExperimentOnlineClusterNSGA3
from pymoo.algorithms.experiment_offline_cluster_nsga3 import ExperimentOfflineClusterNSGA3
from pymoo.model.population import Population
from pymoo.model.evaluator import Evaluator

def generate_transformation_matrix(seed, problem, ref_dirs):
    algorithm = AggregatedGeneticAlgorithm(seed=1)
    algorithm.evaluator = Evaluator()
    read_pop = algorithm.get_initial_population_number(seed)
    algorithm.evaluator.eval(problem, read_pop)
    # print(read_pop.get('X'))
    return read_pop.get('F')

def generate_max_min(problem, reference_directions, number_of_executions):
    print('Generating random solutions for normalization...')
    populations = np.concatenate([generate_transformation_matrix(1, problem, reference_directions) for i in range(number_of_executions)])
    return populations.min(axis=0), populations.max(axis=0)

original_dimension = 8
reduced_dimension = 2
interval_of_aggregations = 1
save_data = True
use_normalization=True
use_different_seeds = True
termination_criterion = ('n_gen', 600)
problem = get_problem("dtlz2", n_obj=original_dimension, n_var=10)
number_of_executions = 21
reference_directions = get_reference_directions("das-dennis", original_dimension, n_partitions=12)

normalization_point = generate_max_min(problem, reference_directions, number_of_executions)
print('Values for normalization', normalization_point)

start = time.time()

experiment = ExperimentOnlineClusterNSGA3(ref_dirs=reference_directions,
    min_max_values=normalization_point,
    use_normalization=use_normalization,
    pop_size=100,
    cluster=AgglomerativeClustering,
    save_dir='.\\experiment_results\\OnlinePearsonClusterNSGA3_{}_{}_{}_{}'.format(problem.name(), original_dimension, reduced_dimension, interval_of_aggregations),
    save_data=True,
    number_of_clusters=reduced_dimension,
    interval_of_aggregations=interval_of_aggregations,
    problem=problem,
    number_of_executions=number_of_executions,
    termination=termination_criterion,
    use_random_aggregation=False,    
    verbose=False,
    save_history=True,
    use_different_seeds=use_different_seeds,
    method='pearson')

print('Online Pearson NSGA-III Experiment Run')
experiment.run()
experiment.show_mean_convergence('igd_convergence.txt')
experiment.show_mean_convergence('hv_convergence.txt')
experiment.show_heat_map()

experiment = ExperimentOnlineClusterNSGA3(ref_dirs=reference_directions,
    min_max_values=normalization_point,
    use_normalization=use_normalization,
    pop_size=100,
    cluster=AgglomerativeClustering,
    save_dir='.\\experiment_results\\OnlineKendallClusterNSGA3_{}_{}_{}_{}'.format(problem.name(), original_dimension, reduced_dimension, interval_of_aggregations),
    save_data=True,
    number_of_clusters=reduced_dimension,
    interval_of_aggregations=interval_of_aggregations,
    problem=problem,
    number_of_executions=number_of_executions,
    termination=termination_criterion,
    use_random_aggregation=False,    
    verbose=False,
    save_history=True,
    use_different_seeds=use_different_seeds,
    method='kendall')

print('Online Kendall NSGA-III Experiment Run')
experiment.run()
experiment.show_mean_convergence('igd_convergence.txt')
experiment.show_mean_convergence('hv_convergence.txt')
experiment.show_heat_map()

experiment = ExperimentNSGA3(ref_dirs=reference_directions,
    min_max_values=normalization_point,
    use_normalization=use_normalization,
    pop_size=100,
    cluster=AgglomerativeClustering,
    save_dir='.\\experiment_results\\NSGA3_{}_{}'.format(problem.name(), original_dimension),
    save_data=True,
    number_of_clusters=reduced_dimension,
    interval_of_aggregations=interval_of_aggregations,
    problem=problem,
    number_of_executions=number_of_executions,
    termination=termination_criterion,
    verbose=False,
    save_history=True,
    use_different_seeds=use_different_seeds)

print('NSGA-III Experiment Run')
experiment.run()
experiment.show_mean_convergence('igd_convergence.txt')
experiment.show_mean_convergence('hv_convergence.txt')

experiment = ExperimentOnlineClusterNSGA3(ref_dirs=reference_directions,
    min_max_values=normalization_point,
    use_normalization=use_normalization,
    pop_size=100,
    cluster=AgglomerativeClustering,
    save_dir='.\\experiment_results\\RandomClusterPearsonNSGA3_{}_{}_{}_{}'.format(problem.name(), original_dimension, reduced_dimension, interval_of_aggregations),
    save_data=True,
    number_of_clusters=reduced_dimension,
    interval_of_aggregations=interval_of_aggregations,
    problem=problem,
    number_of_executions=number_of_executions,
    termination=termination_criterion,
    use_random_aggregation=True,    
    verbose=False,
    save_history=True,
    use_different_seeds=use_different_seeds,
    method='pearson')

print('Random Cluster Pearson NSGA-III Experiment Run')
experiment.run()
experiment.show_mean_convergence('igd_convergence.txt')
experiment.show_mean_convergence('hv_convergence.txt')
experiment.show_heat_map()

experiment = ExperimentOnlineClusterNSGA3(ref_dirs=reference_directions,
    min_max_values=normalization_point,
    use_normalization=use_normalization,
    pop_size=100,
    cluster=AgglomerativeClustering,
    save_dir='.\\experiment_results\\RandomClusterKendallNSGA3_{}_{}_{}_{}'.format(problem.name(), original_dimension, reduced_dimension, interval_of_aggregations),
    save_data=True,
    number_of_clusters=reduced_dimension,
    interval_of_aggregations=interval_of_aggregations,
    problem=problem,
    number_of_executions=number_of_executions,
    termination=termination_criterion,
    use_random_aggregation=True,    
    verbose=False,
    save_history=True,
    use_different_seeds=use_different_seeds,
    method='kendall')

print('Random Cluster Kendall NSGA-III Experiment Run')
experiment.run()
experiment.show_mean_convergence('igd_convergence.txt')
experiment.show_mean_convergence('hv_convergence.txt')
experiment.show_heat_map()

experiment = ExperimentOfflineClusterNSGA3(ref_dirs=reference_directions,
    min_max_values=normalization_point,
    use_normalization=use_normalization,
    pop_size=100,
    cluster=AgglomerativeClustering,
    save_dir='.\\experiment_results\\OfflineClusterPearsonNSGA3_{}_{}_{}'.format(problem.name(), original_dimension, reduced_dimension),
    save_data=True,
    number_of_clusters=reduced_dimension,
    interval_of_aggregations=interval_of_aggregations,
    problem=problem,
    number_of_executions=number_of_executions,
    termination=termination_criterion,
    use_random_aggregation=False,    
    verbose=False,
    save_history=True,
    use_different_seeds=use_different_seeds,
    method='pearson')

print('Offline Cluster NSGA-III Experiment Run')
experiment.run()
experiment.show_mean_convergence('igd_convergence.txt')
experiment.show_mean_convergence('hv_convergence.txt')
experiment.show_heat_map()

end = time.time()
print('Elapsed Time in experiment', end - start)