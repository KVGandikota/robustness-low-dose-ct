import numpy as np

from dival import TaskTable
from dival.measure import PSNR, SSIM

from dliplib.utils import Params
from dliplib.utils.helper import select_hyper_best_parameters, load_standard_dataset
from dliplib.reconstructors import TVAdamReconstructor
from dliplib.utils.reports import save_results_table


# load data
dataset = load_standard_dataset('ellipses')
test_data = dataset.get_data_pairs('validation', 35)

task_table = TaskTable()

# create the reconstructor
reconstructor = TVAdamReconstructor(dataset.ray_trafo)

# create a Dival task table and run it
task_table.append(reconstructor=reconstructor, measures=[PSNR, SSIM],
                  test_data=test_data,
                  hyper_param_choices={'gamma': np.logspace(-4, -2, 10),
                                        'iterations': [2000, 2500, 3000, 3500, 4000, 4500, 5000],
                                        'loss_function': ['mse']})

results = task_table.run()

save_results_table(results, 'ellipses_tvadam')

# select the best hyper-parameters and save them
best_choice, best_error = select_hyper_best_parameters(results)

print(results.to_string(show_columns=['misc']))
print(best_choice)
print(best_error)

params = Params(best_choice)
params.save('ellipses_tvadam')