from run_methods import run_experiment
import sys

max_n_jobs = 1
experiment_type = sys.argv[1]
if len(sys.argv[2]) > 1:
    max_n_jobs = int(sys.argv[2])

if experiment_type == 'AL':
    if __name__ == '__main__':
        run_experiment(
            max_n_jobs=max_n_jobs,
            model_list=['lr','rf','xgb','dnn','df'],
            acquisition_function_list=["random", "exploration", "exploitation", "dynamic","dynamicbald","bald", "similarity"],
            cell_line_list=['A549','Hela','Jurkat','K562'],
            scores_list=['sensitive'],
            feature_type_list=['function','protein'],
            dir_name='results_AL',
            positive_labels_list=[0,1],
            n_start_list=[16,32,64,128,256],
            batch_size_list=[16,32,64],
            random_seed_list=[42,43,44,45,46],
            save_genes=False,
            cold_start=False,
            one_cycle=False,
            max_cycle=5,
            ensemble_size=10,
            learning_rate=0.01,
            dropout_rate=0.2,
            epochs=10,
            cycle_to_draw=5
        )

if experiment_type == 'one_cycle':
    if __name__ == '__main__':
        run_experiment(
            max_n_jobs=max_n_jobs,
            model_list=['lr','rf','xgb','dnn','df'],
            acquisition_function_list=["random", "exploration", "exploitation", "bald", "similarity"],
            cell_line_list=['A549','Hela','Jurkat','K562'],
            scores_list=['sensitive'],
            feature_type_list=['function','protein'],
            dir_name='results_one_cycle',
            positive_labels_list=[0,1],
            n_start_list=[64,128,256],
            batch_size_list=[32,64],
            random_seed_list=[42,43,44,45,46],
            save_genes=False,
            cold_start=False,
            one_cycle=True,
            max_cycle=5,
            ensemble_size=10,
            learning_rate=0.01,
            dropout_rate=0.2,
            epochs=10,
            cycle_to_draw=5
        )

if experiment_type == 'one_cycle_save_genes':
    if __name__ == '__main__':
        run_experiment(
            max_n_jobs=max_n_jobs,
            model_list=['lr','rf','xgb','dnn','df'],
            acquisition_function_list=["random", "exploration", "exploitation", "bald", "similarity"],
            cell_line_list=['A549','Hela','Jurkat','K562'],
            scores_list=['sensitive'],
            feature_type_list=['function','protein'],
            dir_name='results_one_cycle_save_genes',
            positive_labels_list=[0,1],
            n_start_list=[64,128,256],
            batch_size_list=[32,64],
            random_seed_list=[42],
            save_genes=True,
            cold_start=False,
            one_cycle=True,
            max_cycle=5,
            ensemble_size=10,
            learning_rate=0.01,
            dropout_rate=0.2,
            epochs=10,
            cycle_to_draw=5
        )
if experiment_type == 'AL_save_genes':
    if __name__ == '__main__':
        run_experiment(
            max_n_jobs=max_n_jobs,
            model_list=['lr', 'rf', 'xgb', 'dnn', 'df'],
            acquisition_function_list=["random", "exploration", "exploitation", "dynamic", "dynamicbald", "bald","similarity"],
            cell_line_list=['A549', 'Hela', 'Jurkat', 'K562'],
            scores_list=['sensitive'],
            feature_type_list=['function', 'protein'],
            dir_name='results_AL_save_genes',
            positive_labels_list=[0, 1],
            n_start_list=[64, 128, 256],
            batch_size_list=[32,64],
            random_seed_list=[42, 43, 44, 45, 46],
            save_genes=True,
            cold_start=False,
            one_cycle=False,
            max_cycle=5,
            ensemble_size=10,
            learning_rate=0.01,
            dropout_rate=0.2,
            epochs=10,
            cycle_to_draw=5
        )
if experiment_type == 'cold_start':
    if __name__ == '__main__':
        run_experiment(
            max_n_jobs=max_n_jobs,
            model_list=['lr', 'rf', 'xgb', 'dnn', 'df'],
            acquisition_function_list=["random", "exploration", "exploitation", "dynamic", "dynamicbald", "bald",
                                       "similarity"],
            cell_line_list=['A549', 'Hela', 'Jurkat', 'K562'],
            scores_list=['sensitive'],
            feature_type_list=['function', 'protein'],
            dir_name='results_cold_start',
            positive_labels_list=[0, 1],
            n_start_list=[16, 32, 64, 128, 256],
            batch_size_list=[16, 32, 64],
            random_seed_list=[42, 43, 44, 45, 46],
            save_genes=False,
            cold_start=True,
            one_cycle=False,
            max_cycle=5,
            ensemble_size=10,
            learning_rate=0.01,
            dropout_rate=0.2,
            epochs=10,
            cycle_to_draw=5
        )