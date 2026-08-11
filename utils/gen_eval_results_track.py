import os
import pandas as pd


def merge_results(base_path, exp_name, scenes, iteration):
    all_results = pd.DataFrame()
    
    for scene in scenes:
        scene_path = os.path.join(base_path, scene)
        exp_path = os.path.join(scene_path, exp_name)
        result_file = os.path.join(exp_path, f'train/ours_{iteration}/track_metrics.csv')
        
        if not os.path.isdir(scene_path) or not os.path.isfile(result_file):
            print(f"Skipping {scene}: path or result file not found")
            continue
        results = pd.read_csv(result_file, index_col=0) 
        results.columns = [scene]
        all_results = pd.concat([all_results, results], axis=1)
    return all_results


def merge_results_with_stats(base_path, exp_name, scenes, iteration):
    df = merge_results(base_path, exp_name, scenes, iteration)
    count = df.count(axis=1)
    df['Mean'] = df.mean(axis=1)
    df['Std'] = df.std(axis=1)
    df['Min'] = df.min(axis=1)
    df['Max'] = df.max(axis=1)
    df['Count'] = count
    return df


if __name__ == '__main__':
    iteration = '20000'
    # exp_name = 'base_marble'
    exp_name = 'base_mi_nt'
    # exp_name = 'base_tl0.5'
    print(f'Collecting results for {exp_name} at iteration {iteration}')
    dataset = 'artgs' 
    data_path = f'./outputs/{dataset}/sapien'
    scenes = sorted(os.listdir(f'./data/{dataset}/sapien'))
    scenes = [s for s in scenes if 'new' in s and os.path.isdir(os.path.join(data_path, s))]
    df = merge_results_with_stats(data_path, exp_name, scenes, iteration)
    df = df.transpose()
    df.to_csv(f'./outputs/{exp_name}_{iteration}_{dataset}.csv')

    # dataset = 'v2a'
    # data_path = f'./outputs/{dataset}/sapien'
    # scenes = sorted(os.listdir(f'./data/{dataset}/sapien'))
    # scenes = [s for s in scenes if '12071' not in s and '103351' not in s] # 12071 and 103351 have large errors in the data, removed 
    # print("' '".join(scenes))

    # df, df_revolute, df_prismatic, df_revolute_v2a, df_prismatic_v2a, df_all = merge_results_with_stats_v2a(data_path, exp_name, scenes, iteration)
    # df = df.transpose()
    # df.to_csv(f'./outputs/{exp_name}_{iteration}_{dataset}_all.csv')
    # df_revolute = df_revolute.transpose()
    # df_revolute.to_csv(f'./outputs/{exp_name}_{iteration}_{dataset}_revolute.csv')
    # df_prismatic = df_prismatic.transpose()
    # df_prismatic.to_csv(f'./outputs/{exp_name}_{iteration}_{dataset}_prismatic.csv')
    # df_revolute_v2a = df_revolute_v2a.transpose()
    # df_revolute_v2a.to_csv(f'./outputs/{exp_name}_{iteration}_{dataset}_revolute_v2a.csv')
    # df_prismatic_v2a = df_prismatic_v2a.transpose()
    # df_prismatic_v2a.to_csv(f'./outputs/{exp_name}_{iteration}_{dataset}_prismatic_v2a.csv')
    # df_all = df_all.transpose()
    # df_all.to_csv(f'./outputs/{exp_name}_{iteration}_{dataset}_all_v2a.csv')