#此脚本用于跑各种实验，在运行前请确保脚本根目录下有active_learning和active_learning_cold_start两个库及所有需要的数据
import os
#防止计算库底层开多线程引发死锁
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import pandas as pd
import itertools
from concurrent.futures import ProcessPoolExecutor,as_completed
from tqdm import tqdm


#定义主动学习的函数，能够接受config输入，并且能够自动遍历config里的条件
#process_num用于确保不会有两个进程同时写入一个文件，本身数字无意义，但对每个进程必须独一
#这不是配置区，不要随便改
def run_one_task(config,n_jobs_per_process,process_num):
    all_histories = []
    for model in config['model']:
        for acquisition_function in config['acquisition_function']:
            for cell_line in config['cell_line']:
                for scores in config['scores']:
                    for feature_type in config['feature_type']:
                        for positive_label in config['positive_label']:
                            for n_start in config['n_start']:
                                for batch_size in config['batch_size']:
                                    for random_seed in config['random_seed']:
                                        #获取文件名，请确保使用的是依据“scores_cellline_featuretype”格式正确命名的数据文件
                                        BASE_DIR = config['BASE_DIR']
                                        DATA_FILE = os.path.join(BASE_DIR,f'{scores}_{cell_line}_{feature_type}.csv')
                                        feature_cols = [c for c in pd.read_csv(DATA_FILE).columns if c not in ["Gene A","Gene B","label", "y"]]
                                        #处理similarity获取函数名字问题
                                        if feature_type == 'function' and acquisition_function == 'similarity':
                                            acquisition_function = 'similarity_jaccard'
                                        if feature_type == 'protein' and acquisition_function == 'similarity':
                                            acquisition_function = 'similarity_cosine'
                                        # 区分cold_start和正常启动
                                        if config['cold_start']:
                                            from active_learning_cold_start.experiment import Experiment
                                        else:
                                            from active_learning.experiment import Experiment
                                        #处理pool_ratio的逻辑
                                        if cell_line == 'Jurkat' or cell_line == 'K562':
                                            pool_ratio = 0.1
                                        else:
                                            pool_ratio = 1
                                        #处理feature_type的问题
                                        feature_type_for_exp = 'int8'
                                        if feature_type == 'protein':
                                            feature_type_for_exp = 'float32'
                                        #处理one_cycle的逻辑，因为参数不一样所以需要单独拿出来
                                        if config['one_cycle']:
                                            batch_size = batch_size * config['cycle_to_draw'] #这里需要根据实际画图用了多少轮修改
                                            exp = Experiment(
                                                file_path=DATA_FILE,
                                                feature_cols=feature_cols,
                                                acquisition_method=acquisition_function,
                                                seed=random_seed,
                                                batch_size=batch_size,
                                                n_start=n_start,
                                                model_type=model,
                                                model_params=dict(ensemble_size=config['ensemble_size']),
                                                learning_rate=config['learning_rate'],
                                                dropout_rate=config['dropout_rate'],
                                                epochs=config['epochs'],
                                                positive_label=positive_label,
                                                feature_type=feature_type_for_exp,
                                                save_genes=config['save_genes'],
                                                pool_ratio=pool_ratio,
                                                n_jobs=n_jobs_per_process,
                                            )
                                            if config["save_genes"]:
                                                history,genes = exp.run(max_cycles=1)
                                            else:
                                                history = exp.run(max_cycles=1)
                                        else:
                                            exp = Experiment(
                                                file_path=DATA_FILE,
                                                feature_cols=feature_cols,
                                                acquisition_method=acquisition_function,
                                                seed=random_seed,
                                                batch_size=batch_size,
                                                n_start=n_start,
                                                model_type=model,
                                                model_params=dict(ensemble_size=config['ensemble_size']),
                                                learning_rate=config['learning_rate'],
                                                dropout_rate=config['dropout_rate'],
                                                epochs=config['epochs'],
                                                positive_label=positive_label,
                                                feature_type=feature_type_for_exp,
                                                save_genes=config['save_genes'],
                                                pool_ratio=pool_ratio,
                                                n_jobs=n_jobs_per_process,
                                            )
                                            if config["save_genes"]:
                                                history,genes = exp.run(max_cycles=config['max_cycle'])
                                            else:
                                                history = exp.run(max_cycles=config['max_cycle'])
                                        #exp.run本身输出结果就是有[cycle,train_size,pool_size,n_positive,n_start,pos_neg_ratio]
                                        #再额外添加一些信息
                                        history['model'] = model
                                        history['acquisition_function'] = acquisition_function
                                        history['cell_line'] = cell_line
                                        history['scores'] = scores
                                        history['feature_type'] = feature_type
                                        history['positive_label'] = positive_label
                                        history['batch_size'] = batch_size
                                        history['random_seed'] = random_seed
                                        #合并到总结果里
                                        all_histories.append(history)
                                        #如果有save_genes的话，也添加一些信息
                                        if config["save_genes"]:
                                            genes['cell_line'] = cell_line
                                            genes['scores'] = scores
                                            genes['feature_type'] = feature_type
                                            genes['random_seed'] = random_seed
                                            genes.to_csv(os.path.join(RESULT_DIR,f'result_genes_temp{process_num}.csv'),
                                                         index=False,
                                                         header=False,
                                                         mode='a')
    df_history_all = pd.concat(all_histories,ignore_index=True)
    df_history_all.to_csv(os.path.join(RESULT_DIR,f'result_temp{process_num}.csv'),index=False,header=False)

#防止进程递归创建
if __name__ == '__main__':
    #以下为参数配置区————————————————————————————————————————————————————————————————————————————————————————————————————
    #！！！！！运行前请务必反复检查参数是否配置正确,请从给出的备选项里选择参数填入配置
    #！！！！！请不要修改变量类型，若某参数只跑一个值也请保留list形式
    #默认使用并发池增加运行速度

    max_n_jobs = 1 #最多同时使用的cpu核心数，在我们的服务器上绝对不能超过40，最好不要超过三十，和其他人共用服务器时适当减少；dnn需要单独跑，跑dnn时请设置此参数为1
    model_list = ['dnn'] #使用什么模型跑代码，备选包括['lr','rf','xgb','dnn','df','svm']
    #使用何种获取函数，备选项包括["random", "exploration", "exploitation", "dynamic", "dynamicbald", "bald", "similarity"]
    acquisition_function_list = ["random", "exploration", "exploitation", "bald", "similarity"]
    cell_line_list = ['A549','Hela','Jurkat','K562']#跑哪些细胞系，备选项包括['A549','Hela','Jurkat','K562']
    scores_list = ['sensitive']#标签数据中的打分标准，备选包括['sensitive','strong']
    feature_type_list = ['function','protein']#特征数据类型，备选包括['function','protein']
    dir_name = 'result_one_cycle_all_acf_only_dnn' #结果将储存在这个文件夹下，理论上可自由填写
    positive_labels_list = [0,1]#预测的正标签，备选包括[0,1]分别对应SL和SR
    n_start_list = [64,128,256]#初始训练集大小，理论上可自由填写
    batch_size_list = [32,64]#每轮捕获的样本数量，理论上可自由填写
    random_seed_list = [42,43,44,45,46]#随机种子，理论上可自由填写
    #pool_ratio = 1#为了节省算力，在大细胞系里我们会考虑降低这个值，但后面会有单独逻辑判定，一般情况下此参数保持注释即可
    save_genes = False #如果为True，则会额外生成一个文件记录每一轮获取的样本名称，否则只记录获取的样本数量
    cold_start = False #如果为True，则会以冷启动方式启动，即在初始训练集出现过的样本不会在样本池中出现
    one_cycle = True #如果为Ture，则模型会一次性学习one_cycle_num轮主动学习同样数量的样本并进行预测，作为对照组
    max_cycle = 7 #最大学习轮数
    ensemble_size = 10 #模型集群里有多少个不同随机初始化的模型，一般不需要动
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))#根地址，一般来说不需要修改
    learning_rate = 0.01 #学习率，仅对dnn模型生效
    dropout_rate = 0.2 #模型每轮drop掉的神经元比例，设置较大时可缓解过拟合症状
    epochs = 10 #dnn模型训练的轮数
    cycle_to_draw = 5 #用于one_cycle策略，请填入实际画图用了多少轮，一般不用动
    #以上为参数配置区————————————————————————————————————————————————————————————————————————————————————————————————————



    #以下内容除非完全了解其作用，否则不要随意修改，配置区参数已经足以进行大部分实验
    #创建文件夹
    RESULT_DIR = os.path.join(BASE_DIR,dir_name)
    os.makedirs(RESULT_DIR,exist_ok=True)
    #计算确认哪些条件可以做并行
    min_n_jobs = max_n_jobs // 3
    param_grid = {
        "model": model_list,
        "acquisition_function": acquisition_function_list,
        "cell_line": cell_line_list,
        "scores": scores_list,
        "feature_type": feature_type_list,
        "positive_label": positive_labels_list,
        "n_start": n_start_list,
        "batch_size": batch_size_list,
        "random_seed": random_seed_list,
        "save_genes": save_genes,
        "cold_start":cold_start,
        "one_cycle":one_cycle,
        "max_cycle":max_cycle,
        "ensemble_size": ensemble_size,
        "BASE_DIR": BASE_DIR,
        "learning_rate": learning_rate,
        "dropout_rate": dropout_rate,
        "epochs": epochs,
        "cycle_to_draw": cycle_to_draw,

    }
    #拆分进程参数优先级
    parallel_priority = ['random_seed','batch_size','n_start','positive_label',
                         'scores','feature_type','cell_line']
    parallel_keys = []
    n_processes = 1
    for param in parallel_priority:
        n_processes = n_processes * len(param_grid[param])
        if n_processes <= max_n_jobs:
            parallel_keys.append(param)
        if n_processes > max_n_jobs:
            n_processes = n_processes / len(param_grid[param])
    n_processes = int(n_processes)
    n_jobs_per_process = max_n_jobs // n_processes
    #打印信息
    print('用于并行的参数：',parallel_keys)
    print('并行进程数',n_processes)
    print('每个进程所用核心数',n_jobs_per_process)
    print('总理论逻辑核心使用量',n_processes*n_jobs_per_process)
    #生成并行任务组合parallel_configs，list里每个元素对应一个进程的参数
    parallel_configs = []
    base_config = param_grid.copy()
    for param in parallel_keys:
        del base_config[param]
    parallel_param_lists = [param_grid[keys] for keys in parallel_keys]
    product_params = list(itertools.product(*parallel_param_lists))
    for param in product_params:
        full_config = base_config.copy()
        for parallel_key_idx in range(len(parallel_keys)):
            full_config[parallel_keys[parallel_key_idx]] = [param[parallel_key_idx]]
        parallel_configs.append(full_config)
    #开始跑多线程了
    #确保并行进程数和实际参数量对应
    if n_processes != len(parallel_configs):
        assert ValueError('并行进程数和实际参数量不对应，请检查代码')
    process_num_list = range(len(parallel_configs))
    all_histories = []
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        futures = [
            executor.submit(run_one_task,parallel_configs[i],n_jobs_per_process,i)
            for i in process_num_list
        ]
    #进度条
    with tqdm(total=len(futures),desc='总进度') as pbar:
        for future in as_completed(futures):
            #输出结果
            future.result()
            pbar.update(1)
    #把temp文件夹拼接起来
    pbar_concat = tqdm(total=len(process_num_list) + 2,desc='数据后处理进度')
    result_history = pd.DataFrame()
    result_genes = pd.DataFrame()
    for process_num in process_num_list:
        result_history_temp = pd.read_csv(os.path.join(RESULT_DIR,f'result_temp{process_num}.csv'),header=None)
        result_history = pd.concat([result_history,result_history_temp],ignore_index=True,axis=0)
        if save_genes:
            result_genes_temp = pd.read_csv(os.path.join(RESULT_DIR,f'result_genes_temp{process_num}.csv'),header=None)
            result_genes = pd.concat([result_genes,result_genes_temp],ignore_index=True,axis=0)
        pbar_concat.update(1)
    #保存最后的结果，顺便把列名加上
    header_result_history = ['cycle','train_size','pool_size','n_positive','n_start','pos_neg_ratio','model',
                             'acquisition_function','cell_line','scores','feature_type','positive_label',
                             'batch_size','random_seed']
    result_history.to_csv(os.path.join(RESULT_DIR,'result_history.csv'),index=False,header=header_result_history)
    if save_genes:
        header_result_genes = ['Gene_A','Gene_B','label','cycle','n_start','batch_size','positive_label','model','acquisition_function',
                               'cell_line','scores','feature_type','random_seed']
        result_genes.to_csv(os.path.join(RESULT_DIR,'result_genes.csv'),index=False,header=header_result_genes)
    #更新进度条
    pbar_concat.update(1)
    for process_num in process_num_list:
        path_result_history_temp = os.path.join(RESULT_DIR,f'result_temp{process_num}.csv')
        os.remove(path_result_history_temp)
        if save_genes:
            path_result_genes_temp = os.path.join(RESULT_DIR,f'result_genes_temp{process_num}.csv')
            os.remove(path_result_genes_temp)
    pbar_concat.update(1)
