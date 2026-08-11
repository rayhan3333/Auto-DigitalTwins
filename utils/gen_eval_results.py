import os
import pandas as pd
import json


def merge_results(base_path, exp_name, scenes, iteration):
    all_results = pd.DataFrame()
    
    for scene in scenes:
        scene_path = os.path.join(base_path, scene)
        exp_path = os.path.join(scene_path, exp_name)
        result_file = os.path.join(exp_path, f'train/ours_{iteration}/result.csv')
        
        if not os.path.isdir(scene_path) or not os.path.isfile(result_file):
            print(f"Skipping {scene}: path or result file not found")
            continue
        results = pd.read_csv(result_file, index_col=0) 
        results.columns = [scene]
        all_results = pd.concat([all_results, results], axis=1)
    return all_results


def merge_results_v2a(base_path, exp_name, scenes, iteration):
    all_results_revolute = pd.DataFrame()
    all_results_prismatic = pd.DataFrame()
    all_results = pd.DataFrame()
    v2a_results_revolute = pd.DataFrame()
    v2a_results_prismatic = pd.DataFrame()
    v2a_results_all = pd.DataFrame()
    
    v2a_results_dict = json.load(open(os.path.join('results.json'), 'r'))
    for scene in scenes:
        scene_path = os.path.join(base_path, scene)
        exp_path = os.path.join(scene_path, exp_name)
        result_file = os.path.join(exp_path, f'train/ours_{iteration}/result.csv')
        
        if not os.path.isdir(scene_path) or not os.path.isfile(result_file):
            print(f"Skipping {scene}: path or result file not found")
            continue
        results = pd.read_csv(result_file, index_col=0) 
        results.columns = [scene]
        joint_info_file = os.path.join(exp_path, f'train/ours_{iteration}/joint_info.json')
        joint_info = json.load(open(joint_info_file, 'r'))
        joint_type = joint_info[1]['joint']
        all_results = pd.concat([all_results, results], axis=1)
        v2a_result = pd.DataFrame()
        result_dict = v2a_results_dict[f'{scene}_']
        v2a_result['CD_static'] = [result_dict['geometry_error']['static_chamfer_distance']]
        v2a_result['CD_whole'] = [result_dict['geometry_error']['full_chamfer_distance']]
        v2a_result['CD_dynamic'] = [result_dict['geometry_error']['moving_chamfer_distance']]
        v2a_result['angle'] = [result_dict['joint_error']['joint orientation error']]
        v2a_result['distance'] = [result_dict['joint_error']['joint position error']]
        v2a_result['joint_state_error'] = [result_dict['joint_error']['joint state error']]
        if v2a_result['CD_static'][0] == 1.0:
            v2a_result['CD_static'] = [100]
            v2a_result['CD_whole'] = [100]
            v2a_result['CD_dynamic'] = [100]
            v2a_result['angle'] = [90]
            v2a_result['distance'] = [100]
            v2a_result['joint_state_error'] = [90]
        v2a_result = v2a_result.transpose()
        v2a_result.columns = [scene]
        v2a_results_all = pd.concat([v2a_results_all, v2a_result], axis=1)
        if joint_type == 'hinge':
            all_results_revolute = pd.concat([all_results_revolute, results], axis=1)
            v2a_results_revolute = pd.concat([v2a_results_revolute, v2a_result], axis=1)
        elif joint_type == 'slider':
            all_results_prismatic = pd.concat([all_results_prismatic, results], axis=1)
            v2a_results_prismatic = pd.concat([v2a_results_prismatic, v2a_result], axis=1)
        else:
            raise ValueError(f"Skipping {scene}: joint type {joint_type} not supported")
    return all_results, all_results_revolute, all_results_prismatic, v2a_results_revolute, v2a_results_prismatic, v2a_results_all


def merge_results_with_stats(base_path, exp_name, scenes, iteration):
    df = merge_results(base_path, exp_name, scenes, iteration)
    count = df.count(axis=1)
    df['Mean'] = df.mean(axis=1)
    df['Std'] = df.std(axis=1)
    df['Min'] = df.min(axis=1)
    df['Max'] = df.max(axis=1)
    df['Count'] = count
    return df


def merge_results_with_stats_v2a(base_path, exp_name, scenes, iteration):
    df, df_revolute, df_prismatic, df_revolute_v2a, df_prismatic_v2a, df_all = merge_results_v2a(base_path, exp_name, scenes, iteration)
    count = df.count(axis=1)
    df['Mean'] = df.mean(axis=1)
    df['Std'] = df.std(axis=1)
    df['Min'] = df.min(axis=1)
    df['Max'] = df.max(axis=1)
    df['Count'] = count
    count_revolute = df_revolute.count(axis=1)
    count_prismatic = df_prismatic.count(axis=1)
    df_revolute['Mean'] = df_revolute.mean(axis=1)
    df_revolute['Std'] = df_revolute.std(axis=1)
    df_revolute['Min'] = df_revolute.min(axis=1)
    df_revolute['Max'] = df_revolute.max(axis=1)
    df_revolute['Count'] = count_revolute
    df_prismatic['Mean'] = df_prismatic.mean(axis=1)
    df_prismatic['Std'] = df_prismatic.std(axis=1)
    df_prismatic['Min'] = df_prismatic.min(axis=1)
    df_prismatic['Max'] = df_prismatic.max(axis=1)
    df_prismatic['Count'] = count_prismatic
    df_revolute_v2a['Mean'] = df_revolute_v2a.mean(axis=1)
    df_revolute_v2a['Std'] = df_revolute_v2a.std(axis=1)
    df_revolute_v2a['Min'] = df_revolute_v2a.min(axis=1)
    df_revolute_v2a['Max'] = df_revolute_v2a.max(axis=1)
    df_revolute_v2a['Count'] = count_revolute
    df_prismatic_v2a['Mean'] = df_prismatic_v2a.mean(axis=1)
    df_prismatic_v2a['Std'] = df_prismatic_v2a.std(axis=1)
    df_prismatic_v2a['Min'] = df_prismatic_v2a.min(axis=1)
    df_prismatic_v2a['Max'] = df_prismatic_v2a.max(axis=1)
    df_prismatic_v2a['Count'] = count_prismatic
    df_all['Mean'] = df_all.mean(axis=1)
    df_all['Std'] = df_all.std(axis=1)
    df_all['Min'] = df_all.min(axis=1)
    df_all['Max'] = df_all.max(axis=1)
    df_all['Count'] = count_revolute + count_prismatic
    return df, df_revolute, df_prismatic, df_revolute_v2a, df_prismatic_v2a, df_all


if __name__ == '__main__':
    iteration = '20000'
    exp_name = 'final'
    print(f'Collecting results for {exp_name} at iteration {iteration}')
    dataset = 'videoartgs' 
    data_path = f'/mnt/fillipo/yuliu/VideoArtGS/outputs/{dataset}/sapien'
    scenes = sorted(os.listdir(f'./data/{dataset}/sapien'))
    scenes = [s for s in scenes if os.path.isdir(os.path.join(data_path, s))]
    df = merge_results_with_stats(data_path, exp_name, scenes, iteration)
    df = df.transpose()
    df.to_csv(f'/mnt/fillipo/yuliu/VideoArtGS/outputs/{exp_name}_{iteration}_{dataset}.csv')

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