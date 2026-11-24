# -*- coding: utf-8 -*-
import hashlib
import json
import os
import platform
import traceback
from importlib.metadata import version, PackageNotFoundError
import numpy as np
import pandas as pd
from active_learning.experiment import Experiment

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(BASE_DIR, 'results_custom')
CELLS = ['A549']
FEATURES = ['function']
LABEL_TYPES = ['sensitive']
THRESHOLDS = [10]
TASKS = ['SL']
MODELS = ['df','dnn']
METHODS = ['random','exploitation','exploration','dynamic','bald','dynamicbald','similarity','frequency_rank','frequency_value']
SEEDS = list(range(42,52))
N_START_LIST = [64]
BATCH_SIZE = 32
MAX_CYCLES = 5
MAX_SCREEN_SIZE = None
ENSEMBLE_SIZE = 10
N_JOBS = 8
LEARNING_RATE = 0.01
DROPOUT_RATE = 0.2
EPOCHS = 10
RESUME_COMPLETED = True
DTYPES = {'function': 'int8', 'protein': 'float32'}
POSITIVE_LABELS = {'SL': 0, 'SR': 1}
VALID_METHODS = {'random','exploration','exploitation','dynamic','dynamicbald','bald', 'frequency_rank','frequency_value','similarity','similarity_jaccard','similarity_cosine'}

def methods_for_feature(feature):
    result = []
    for name in METHODS:
        value = ('similarity_jaccard' if feature == 'function' else 'similarity_cosine') if name == 'similarity' else name
        if (feature == 'function' and value == 'similarity_cosine') or (feature == 'protein' and value == 'similarity_jaccard'):
            raise ValueError(f'{feature}与获取策略{value}不匹配；请使用similarity别名或正确的相似性策略')
        if value not in result: result.append(value)
    return result

def validate_settings():
    choices = [('CELLS',CELLS,{'A549','HeLa','K562','Jurkat'}),
               ('FEATURES',FEATURES,set(DTYPES)),('LABEL_TYPES',LABEL_TYPES,{'sensitive','strong'}), ('TASKS',TASKS,set(POSITIVE_LABELS)),('MODELS',MODELS,{'lr','rf','xgb','df','dnn'}), ('METHODS',METHODS,VALID_METHODS)]
    for name, values, allowed in choices:
        if not values or len(values) != len(set(values)) or set(values)-allowed:
            raise ValueError(f'{name}为空、包含重复项或无效名称：{values}；允许值={sorted(allowed)}')
    for name, values in [('THRESHOLDS',THRESHOLDS),('SEEDS',SEEDS),('N_START_LIST',N_START_LIST)]:
        if not values or len(values) != len(set(values)) or any(not isinstance(x, (int,np.integer)) for x in values):
            raise ValueError(f'{name}必须为非空且不重复的整数列表')
    if any(x not in (5,10,15,20,50,100) for x in THRESHOLDS): raise ValueError('THRESHOLDS只支持5、10、15、20、50、100')
    if any(x < 2 for x in N_START_LIST) or any(x < 0 for x in SEEDS): raise ValueError('初始样本数需>=2、seed需>=0')
    if BATCH_SIZE < 1 or MAX_CYCLES < 1 or ENSEMBLE_SIZE < 1 or N_JOBS < 1: raise ValueError('批大小、轮数、集成数量、线程数必须>=1')
    for feature in FEATURES: methods_for_feature(feature)

def input_path(cell, feature, label_type, pct):
    return os.path.join(BASE_DIR, f'{cell}_{label_type}_{pct:03d}pct_{feature}.csv')

def data_output_dir(root, cell, feature, label_type, pct, task):
    return os.path.join(root, feature, cell, label_type, f'{pct:03d}pct', task)

def run_paths(data_dir, input_file, task, model_name, method, seed, n_start):
    stem = os.path.splitext(os.path.basename(input_file))[0]
    name = f'{stem}_{task}_{model_name}_{method}_seed{seed}_start{n_start}_batch{BATCH_SIZE}'
    directory = os.path.join(data_dir, 'runs', name)
    return (directory, *(os.path.join(directory, fn) for fn in ['active_learning_history.csv','acquired_gene_pairs.csv','initialization_summary.csv']))

def checksum(path):
    if not os.path.isfile(path): raise FileNotFoundError(f'缺少代码文件：{path}')
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def prepare_output():
    validate_settings()
    input_metadata = {}
    for cell in CELLS:
        for feature in FEATURES:
            for label_type in LABEL_TYPES:
                for pct in THRESHOLDS:
                    path = input_path(cell, feature, label_type, pct)
                    if not os.path.isfile(path): raise FileNotFoundError(f'找不到输入文件：{path}')
                    stat = os.stat(path)
                    input_metadata[os.path.relpath(path,BASE_DIR)] = {'size_bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
    names = [os.path.basename(__file__),'active_learning/experiment.py','active_learning/get_data.py','active_learning/Handler.py', 'active_learning/acquisition.py','active_learning/model.py']
    code_hashes = {name:checksum(os.path.join(BASE_DIR,name)) for name in names}
    packages = {}
    for name in ['numpy','pandas','scikit-learn','torch','xgboost','deep-forest']:
        try: packages[name] = version(name)
        except PackageNotFoundError: packages[name] = 'not-installed'

    config = {'cells':CELLS,'features':FEATURES,'label_types':LABEL_TYPES,'thresholds':THRESHOLDS, 'tasks':TASKS,'models':MODELS,'methods':METHODS,'seeds':SEEDS,'n_start_list':N_START_LIST,
              'batch_size':BATCH_SIZE,'max_cycles':MAX_CYCLES,'max_screen_size':MAX_SCREEN_SIZE, 'ensemble_size':ENSEMBLE_SIZE,'n_jobs':N_JOBS,'learning_rate':LEARNING_RATE,
              'dropout_rate':DROPOUT_RATE,'epochs':EPOCHS,'input_files':input_metadata,'code_hashes':code_hashes, 'python_version':platform.python_version(),'package_versions':packages}
    digest = hashlib.sha256(json.dumps(config,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:12]
    root = os.path.join(RESULTS_ROOT, f'config_{digest}')
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root,'experiment_config.json'),'w',encoding='utf-8') as f:
        json.dump(config,f,indent=2,ensure_ascii=False)
    metadata = {'python':platform.python_version(),'packages':packages,'code_sha256':code_hashes}
    with open(os.path.join(root,'environment_and_code.json'),'w',encoding='utf-8') as f:
        json.dump(metadata,f,indent=2,ensure_ascii=False)
    return root

def initial_pair_hash(acquired):
    pairs = acquired.loc[acquired['cycle'].eq(0), ['Gene A','Gene B']]
    names = sorted(f'{a}\t{b}' for a,b in pairs.itertuples(index=False,name=None))
    return hashlib.sha256('\n'.join(names).encode('utf-8')).hexdigest()

def load_complete_run(paths):
    _, hist_file, pair_file, init_file = paths
    if not all(os.path.isfile(path) for path in (hist_file,pair_file,init_file)): return None
    try:
        hist = pd.read_csv(hist_file)
        pairs = pd.read_csv(pair_file,dtype={'Gene A':str,'Gene B':str},keep_default_na=False)
        init = pd.read_csv(init_file)
        if hist.empty or pairs.empty or init.empty: return None
        needed = {'cycle','train_size','pool_size','n_positive','n_new_positive','n_positive_new_cumulative', 'n_exploit','n_explore','n_start'}
        if not needed.issubset(hist.columns) or not {'cycle','Gene A','Gene B','is_positive','acquisition_source'}.issubset(pairs.columns): return None
        if hist['cycle'].tolist() != list(range(MAX_CYCLES+1)): return None
        if pairs.duplicated(['Gene A','Gene B']).any(): return None
        if len(pairs) != int(hist['train_size'].iloc[-1]): return None
        if int(hist['train_size'].iloc[0]) != int(hist['n_start'].iloc[0]): return None
        if int(pairs['is_positive'].sum()) != int(hist['n_positive'].iloc[-1]): return None
        cumulative = 0
        for _,row in hist.iterrows():
            cyc = int(row['cycle']); selected = pairs.loc[pairs['cycle'].eq(cyc)]
            if cyc == 0:
                if len(selected) != int(row['train_size']) or int(row['n_new_positive']) != 0: return None
            else:
                if len(selected) != int(row['train_size'])-int(hist.iloc[cyc-1]['train_size']): return None
                if int(row['n_new_positive']) != int(selected['is_positive'].sum()): return None
                cumulative += int(row['n_new_positive'])
            if int(row['n_positive_new_cumulative']) != cumulative: return None
        if int(hist['n_positive'].iloc[-1]-hist['n_positive'].iloc[0]) != cumulative: return None
        return hist,pairs,init
    except (ValueError,KeyError,OSError,pd.errors.ParserError,TypeError):
        return None

def summarize(all_hist, data_dir):
    group = ['dataset','cell_line','feature','label_type','threshold_pct','task','model','method','n_start','batch_size','cycle']
    numbers = ['train_size','pool_size','n_positive','n_new_positive','n_positive_new_cumulative', 'n_exploit','n_explore','n_unique_gene_pairs']
    numbers = [x for x in numbers if x in all_hist.columns]
    result = all_hist.groupby(group,dropna=False)[numbers].agg(['mean','std','count']).reset_index()
    result.columns = ['_'.join(k).strip('_') if isinstance(k,tuple) else k for k in result.columns]
    for metric in numbers: result[f'{metric}_se'] = result[f'{metric}_std']/np.sqrt(result[f'{metric}_count'])
    result.to_csv(os.path.join(data_dir,'active_learning_results_mean_sd_se.csv'),index=False,encoding='utf-8-sig')
    wanted = group + [x for x in result.columns if x.startswith(('n_positive_','n_new_positive_','n_positive_new_cumulative_'))]
    result[list(dict.fromkeys(wanted))].to_csv(os.path.join(data_dir,'sl_hits_plot_data_mean_sd_se.csv'),index=False,encoding='utf-8-sig')
    final = all_hist.loc[all_hist['cycle'].eq(MAX_CYCLES)].copy()
    columns = ['dataset','cell_line','feature','label_type','threshold_pct','task','model','method', 'n_start','batch_size','seed','n_positive','n_positive_new_cumulative']
    final[columns].to_csv(os.path.join(data_dir,'final_hits_all_seeds.csv'),index=False,encoding='utf-8-sig')
    final_group = [x for x in group if x!='cycle']
    last = final.groupby(final_group,dropna=False)['n_positive_new_cumulative'].agg(['mean','std','count']).reset_index()
    last['se'] = last['std']/np.sqrt(last['count'])
    last.to_csv(os.path.join(data_dir,'final_new_hits_mean_sd_se.csv'),index=False,encoding='utf-8-sig')
    shortage = last.loc[last['count'].ne(len(SEEDS))]
    path = os.path.join(data_dir,'incomplete_seed_groups.csv')
    if not shortage.empty:
        shortage.to_csv(path,index=False,encoding='utf-8-sig')
        print(f'[WARNING] {len(shortage)}组实验的有效重复次数不足{len(SEEDS)}',flush=True)
    elif os.path.isfile(path): os.remove(path)

def run_dataset(root,cell,feature,label_type,pct,task):
    path = input_path(cell,feature,label_type,pct)
    data_dir = data_output_dir(root,cell,feature,label_type,pct,task)
    os.makedirs(data_dir,exist_ok=True)
    headers = list(pd.read_csv(path,nrows=0,encoding='utf-8-sig').columns)
    if len(headers) != len(set(headers)): raise ValueError(f'特征列名重复：{path}')
    cols = [c for c in headers if c not in ('Gene A','Gene B','label','y')]
    expected = 6810 if feature == 'function' else 2560
    if len(cols) != expected: raise ValueError(f'{path}应包含{expected}列特征，实际{len(cols)}列')
    histories,initials,failures,initial_checks = [],[],[],{}
    for seed in SEEDS:
        for n_start in N_START_LIST:
            for model_name in MODELS:
                for method in methods_for_feature(feature):
                    title = f'{cell}/{feature}/{label_type}/{pct}%/{task} model={model_name} method={method} seed={seed} start={n_start}'
                    paths = run_paths(data_dir,path,task,model_name,method,seed,n_start)
                    try:
                        loaded = load_complete_run(paths) if RESUME_COMPLETED else None
                        if loaded is None:
                            print(f'[RUN] {title}',flush=True)
                            exp = Experiment(file_path=path,feature_cols=cols,test_size=0.0,n_start=n_start, batch_size=BATCH_SIZE,max_screen_size=MAX_SCREEN_SIZE,acquisition_method=method,
                                model_type=model_name,model_params={'ensemble_size':ENSEMBLE_SIZE},seed=seed, n_jobs=N_JOBS,learning_rate=LEARNING_RATE,drop_out_p=DROPOUT_RATE,epochs=EPOCHS,
                                positive_label=POSITIVE_LABELS[task],feature_type=DTYPES[feature],pool_ratio=1.0, output_dir=os.path.join(data_dir,'runs'))
                            exp.run(max_cycles=MAX_CYCLES)
                            loaded = load_complete_run(paths)
                            if loaded is None: raise RuntimeError('结果未写入或轮次、阳性数量、基因对明细未通过校验')
                        else: print(f'[RESUME] {title}',flush=True)
                        hist,acquired,init = loaded

                        key = (seed,n_start)
                        fingerprint = initial_pair_hash(acquired)
                        if key in initial_checks and initial_checks[key] != fingerprint:
                            raise AssertionError('同一seed与n_start下初始基因对不一致')
                        initial_checks[key] = fingerprint
                        hist = hist.copy()
                        hist['dataset'] = os.path.basename(path);hist['cell_line'] = cell
                        hist['feature'] = feature;hist['label_type'] = label_type;hist['threshold_pct'] = pct
                        hist['task'] = task;hist['seed'] = seed;hist['model'] = model_name
                        hist['method'] = method;hist['batch_size'] = BATCH_SIZE
                        hist['init_pair_sha256'] = fingerprint
                        histories.append(hist)
                        info = init.iloc[0].to_dict();info['dataset'] = os.path.basename(path)
                        info['init_pair_sha256'] = fingerprint;initials.append(info)
                    except Exception as exc:
                        print(f'[FAILED] {title}: {exc}',flush=True)
                        failures.append({'dataset':os.path.basename(path),'cell_line':cell,'feature':feature,
                            'label_type':label_type,'threshold_pct':pct,'task':task,'model':model_name, 'method':method,'seed':seed,'n_start':n_start,'error':str(exc), 'traceback':traceback.format_exc(limit=5)})
    if histories:
        df = pd.concat(histories,ignore_index=True).sort_values(['seed','model','method','n_start','cycle'])
        df.to_csv(os.path.join(data_dir,'active_learning_results_all_seeds.csv'),index=False,encoding='utf-8-sig')
        plot_cols = ['dataset','seed','model','method','n_start','cycle','train_size','n_positive', 'n_new_positive','n_positive_new_cumulative','pos_neg_ratio']
        df[plot_cols].to_csv(os.path.join(data_dir,'sl_hits_plot_data_all_seeds.csv'),index=False,encoding='utf-8-sig')
        pd.DataFrame(initials).to_csv(os.path.join(data_dir,'initialization_all_seeds.csv'),index=False,encoding='utf-8-sig')
        summarize(df,data_dir)
    bad_path = os.path.join(data_dir,'failed_runs.csv')
    if failures: pd.DataFrame(failures).to_csv(bad_path,index=False,encoding='utf-8-sig')
    elif os.path.isfile(bad_path): os.remove(bad_path)
    return len(histories), failures

def main():
    root = prepare_output()
    planned = len(CELLS)*len(LABEL_TYPES)*len(THRESHOLDS)*len(TASKS)*len(SEEDS)*len(N_START_LIST)*len(MODELS)*sum(len(methods_for_feature(x)) for x in FEATURES)
    print(f'[CONFIG] 计划独立实验数={planned} | 每次轮数={MAX_CYCLES} | 每轮获取={BATCH_SIZE}',flush=True)
    print(f'[DATA] {BASE_DIR}\n[OUTPUT] {root}',flush=True)
    done,all_failures = 0,[]
    for cell in CELLS:
        for feature in FEATURES:
            for label_type in LABEL_TYPES:
                for pct in THRESHOLDS:
                    for task in TASKS:
                        try:
                            success,failed = run_dataset(root,cell,feature,label_type,pct,task)
                            done += success;all_failures.extend(failed)
                        except Exception as exc:
                            label = f'{cell}/{feature}/{label_type}/{pct}%/{task}'
                            print(f'[DATASET FAILED] {label}: {exc}',flush=True)
                            all_failures.append({'dataset':label,'error':str(exc),'traceback':traceback.format_exc(limit=5)})
    bad_path = os.path.join(root,'all_failures.csv')
    if all_failures:
        pd.DataFrame(all_failures).to_csv(bad_path,index=False,encoding='utf-8-sig')
        raise RuntimeError(f'{len(all_failures)}组失败/异常，完成{done}/{planned}组；请检查{bad_path}')
    if os.path.isfile(bad_path): os.remove(bad_path)
    if done != planned: raise RuntimeError(f'实际完成{done}组，与计划{planned}组不一致')
    print(f'[ALL DONE] {done}/{planned}组独立实验完成，结果位于 {root}',flush=True)

if __name__ == '__main__': main()
