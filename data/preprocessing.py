from glob import glob
import pandas as pd
from utils import make_day_array
from tqdm.auto import tqdm

def make_combination(species, dataframe):
    before_file_path = []
    after_file_path = []
    time_delta = []

    for version in tqdm(dataframe[dataframe['species']==species]['version'].unique()):
        for i in range(len(dataframe[dataframe['version']==version])-1):
            for j in range(len(dataframe[dataframe['version']==version])):
                after = dataframe[dataframe['version']==version].iloc[j].reset_index(drop=True)
                before = dataframe[dataframe['version']==version].iloc[i].reset_index(drop=True)

                if int(after[1]) > int(befor[1]):
                    before_file_path.append(before[0])
                    after_file_path.append(after[0])

                    delta = int(after[1] - before[1])
                    time_delta.append(delta)
    
    combination_df = pd.DataFrame({
        'before_file_path': before_file_path,
        'after_file_path': after_file_path,
        'time_delta': time_delta
    })

    combination_df['species'] = species

    return combination_df

def make_train_df(root_path):
    bc_direct = glob(root_path + '/BC/*')
    bc_direct_name = [x[-5:] for x in bc_direct]
    lt_direct = glob(root_path + '/LT/*')
    lt_direct_name = [x[-5:] for x in lt_direct]

    bc_images = {key: glob(name + '.*.png') for key, name in zip(bc_direct_name, bc_direct)}
    lt_images = {key: glob(name + '/*.png') for key, name in zip(lt_direct_name, lt_direct)}

    bc_days = {key: make_day_array(bc_images[key]) for key in bc_direct_name}
    lt_days = {key: make_day_array(lt_images[key]) for key in lt_direct_name}

    bc_dfs = []

    for i in bc_direct_name:
        bc_df = pd.DataFrame({
            'file_name': bc_images[i],
            'day': bc_days[i],
            'species': 'bc',
            'version': i
        })
        bc_dfs.append(bc_df)

    lt_dfs = []

    for i in lt_direct_name:
        lt_df = pd.DataFrame({
            'file_name': lt_images[i],
            'day': lt_days[i],
            'species': 'lt',
            'version': i
        })
        lt_dfs.append(lt_df)

    bc_dataframe = pd.concat(bc_dfs).reset_index(drop=True)
    lt_dataframe = pd.concat(lt_dfs).reset_index(drop=True)

    total_dataframe = pd.concat([bc_dataframe, lt_dataframe]).reset_index(drop=True)
    # Noise 데이터 제거
    noise_path = [root_path+'/BC/BC_08/DAT02.png',
              root_path+'/BC/BC_08/DAT03.png']
    
    for path in noise_path:
        noise = total_dataframe[total_dataframe['file_name']==path].index
        total_dataframe.drop(noise, inplace=True)

    bc_combination = make_combination('bc', total_dataframe)
    lt_combination = make_combination('lt', total_dataframe)

    total_df = pd.concat([bc_combination, lt_combination])

    return total_df

def make_test_df(test_path):
    test_set = pd.read_csv(test_path+'/test_data.csv')
    test_set['l_root'] = test_set['before_file_path'].map(lambda x:test_path+'/'+x.split('_')[1]+'/'+x.split('_')[2])
    test_set['l_root'] = test_set['after_file_path'].map(lambda x:test_path+'/'+x.split('_')[1]+'/'+x.split('_')[2])

    test_df = pd.DataFrame()
    test_df['before_file_path'] = test_set['l_root'] + '/' + test_set['before_file_path'] + '.png'
    test_df['after_file_path'] = test_set['r_root'] + '/' + test_set['after_file_path'] + '.png'

    return test_df
