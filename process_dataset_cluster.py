import numpy as np
import pandas as pd
import os
import sys
import random
import json
import argparse
from cutmix_intra import cutmix_tabular, cutmix_tabular_cluster
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import chi2_contingency



TYPE_TRANSFORM ={
    'float', np.float32,
    'str', str,
    'int', int
}

INFO_PATH = 'data/Info'

parser = argparse.ArgumentParser(description='process dataset')

# General configs
parser.add_argument('--dataname', type=str, default=None, help='Name of dataset.')
args = parser.parse_args()


def cramers_v(x, y):
    """计算分类变量之间的相关性 Cramér's V"""
    contingency_table = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2corr = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def eta_squared(categorical, numerical):
    """计算分类与数值变量之间的相关性 ETA 系数的平方"""
    groups = categorical.unique()
    mean_total = numerical.mean()
    ss_total = sum((numerical - mean_total) ** 2)
    ss_between = sum(
        len(numerical[categorical == g]) * (numerical[categorical == g].mean() - mean_total) ** 2 for g in groups)
    return ss_between / ss_total


def compute_correlation(train_df):
    """计算所有特征之间的相关性矩阵（数值、分类混合）"""
    columns = train_df.columns
    corr_matrix = pd.DataFrame(np.zeros((len(columns), len(columns))), columns=columns, index=columns)

    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i >= j:  # 对称矩阵，不重复计算
                continue

            if train_df[col1].dtype in ['int64', 'float64'] and train_df[col2].dtype in ['int64', 'float64']:
                # 数值特征之间使用皮尔逊相关系数
                corr = abs(train_df[col1].corr(train_df[col2]))  # 取绝对值
            elif train_df[col1].dtype == 'object' and train_df[col2].dtype == 'object':
                # 分类特征之间使用 Cramér's V
                corr = abs(cramers_v(train_df[col1], train_df[col2]))  # 取绝对值
            else:
                # 数值与分类特征之间使用 ETA 系数
                if train_df[col1].dtype == 'object':
                    corr = np.sqrt(abs(eta_squared(train_df[col1], train_df[col2])))  # 取绝对值
                else:
                    corr = np.sqrt(abs(eta_squared(train_df[col2], train_df[col1])))  # 取绝对值

            corr_matrix.loc[col1, col2] = corr
            corr_matrix.loc[col2, col1] = corr

    return corr_matrix


def preprocess_data(train_df):
    """
    对数据进行预处理，处理分类特征，确保相关性矩阵计算不会出错。

    参数:
    - train_df: pandas DataFrame, 原始数据。

    返回:
    - cleaned_df: pandas DataFrame, 经过预处理的数据。
    """
    # 检查每列数据类型
    for col in train_df.columns:
        if train_df[col].dtype == 'object':
            # 对分类变量进行填充空值处理并编码为字符串
            train_df[col] = train_df[col].fillna("Unknown")
    return train_df


def feature_clustering_mixed(train_df, column_names, threshold=0.8):
    """
    对混合特征（数值和分类）计算相关性，并进行聚类。

    参数:
    - train_df: pandas DataFrame, 输入数据。
    - threshold: float, 聚类的相关性阈值。

    返回:
    - subsets: List, 每个子集包含的特征。
    """
    # 数据预处理
    train_df = preprocess_data(train_df)

    # 计算相关性矩阵
    corr_matrix = compute_correlation(train_df)
    corr_matrix = corr_matrix.clip(-1, 1)  # 强制将相关性限制在 -1 和 1 之间
    print(corr_matrix)
    # 转换为距离矩阵 (1 - 相关性)，用于聚类
    distance_matrix = 1 - corr_matrix.abs()  # 确保所有值为非负数
    # print(distance_matrix)
    condensed_distance = squareform(distance_matrix.values, checks=False)
    # print(condensed_distance)
    # 层次聚类
    linkage_matrix = linkage(condensed_distance, method='average')
    # print(linkage_matrix)
    # 根据阈值进行聚类
    clusters = fcluster(linkage_matrix, t=threshold, criterion='distance')

    # 聚类结果整理
    cluster_dict = {}
    for idx, cluster_id in enumerate(clusters):
        cluster_dict.setdefault(cluster_id, []).append(corr_matrix.columns[idx])

    # 打印每个子集和对应的相关性
    subsets = []
    for cluster_id, features in cluster_dict.items():
        print(f"Cluster {cluster_id}:")
        subsets.append(features)
        for i, feature1 in enumerate(features):
            for feature2 in features[i + 1:]:
                # print(f"Correlation between {column_names[feature1]} and {column_names[feature2]}: {corr_matrix.loc[feature1, feature2]:.2f}")
                print(f"Correlation between {feature1} and {feature2}: {corr_matrix.loc[feature1, feature2]:.2f}")
        print()
    # print(column_names)
    return subsets
def get_column_name_mapping(data_df, num_col_idx, cat_col_idx, target_col_idx, column_names = None):
    
    if not column_names:
        column_names = np.array(data_df.columns.tolist())
    

    idx_mapping = {}

    curr_num_idx = 0
    curr_cat_idx = len(num_col_idx)
    curr_target_idx = curr_cat_idx + len(cat_col_idx)

    for idx in range(len(column_names)):

        if idx in num_col_idx:
            idx_mapping[int(idx)] = curr_num_idx
            curr_num_idx += 1
        elif idx in cat_col_idx:
            idx_mapping[int(idx)] = curr_cat_idx
            curr_cat_idx += 1
        else:
            idx_mapping[int(idx)] = curr_target_idx
            curr_target_idx += 1


    inverse_idx_mapping = {}
    for k, v in idx_mapping.items():
        inverse_idx_mapping[int(v)] = k
        
    idx_name_mapping = {}
    
    for i in range(len(column_names)):
        idx_name_mapping[int(i)] = column_names[i]

    return idx_mapping, inverse_idx_mapping, idx_name_mapping


def train_val_test_split(data_df, cat_columns, num_train = 0, num_test = 0):
    total_num = data_df.shape[0]
    idx = np.arange(total_num)


    seed = 82

    while True:
        np.random.seed(seed)
        np.random.shuffle(idx)

        train_idx = idx[:num_train]
        test_idx = idx[-num_test:]

        train_df = data_df.loc[train_idx]
        test_df = data_df.loc[test_idx]

        flag = 0
        for i in cat_columns:
            if len(set(train_df[i])) != len(set(data_df[i])):
                flag = 1
                break

        if flag == 0:
            break
        else:
            seed += 1
        
    return train_df, test_df, seed


def preprocess_beijing():
    with open(f'{INFO_PATH}/beijing.json', 'r') as f:
        info = json.load(f)

    data_path = info['raw_data_path']

    data_df = pd.read_csv(data_path)
    columns = data_df.columns

    data_df = data_df[columns[1:]]

    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(info['data_path'], index=False)


def preprocess_news():
    with open(f'{INFO_PATH}/news.json', 'r') as f:
        info = json.load(f)

    data_path = info['raw_data_path']
    data_df = pd.read_csv(data_path)
    data_df = data_df.drop('url', axis=1)

    columns = np.array(data_df.columns.tolist())

    cat_columns1 = columns[list(range(12, 18))]
    cat_columns2 = columns[list(range(30, 38))]

    cat_col1 = data_df[cat_columns1].astype(int).to_numpy().argmax(axis=1)
    cat_col2 = data_df[cat_columns2].astype(int).to_numpy().argmax(axis=1)

    data_df = data_df.drop(cat_columns2, axis=1)
    data_df = data_df.drop(cat_columns1, axis=1)

    data_df['data_channel'] = cat_col1
    data_df['weekday'] = cat_col2

    data_save_path = 'data/news/news.csv'
    data_df.to_csv(f'{data_save_path}', index=False)

    columns = np.array(data_df.columns.tolist())
    num_columns = columns[list(range(45))]
    cat_columns = ['data_channel', 'weekday']
    target_columns = columns[[45]]

    info['num_col_idx'] = list(range(45))
    info['cat_col_idx'] = [46, 47]
    info['target_col_idx'] = [45]
    info['data_path'] = data_save_path

    name = 'news'
    with open(f'{INFO_PATH}/{name}.json', 'w') as file:
        json.dump(info, file, indent=4)

def process_data(name):

    with open(f'{INFO_PATH}/{name}.json', 'r') as f:
        info = json.load(f)

    data_path = info['data_path']
    print(data_path)
    if info['file_type'] == 'csv':
        data_df = pd.read_csv(data_path, header = info['header'])

    elif info['file_type'] == 'xls':
        data_df = pd.read_excel(data_path, sheet_name='Data', header=1)
        data_df = data_df.drop('ID', axis=1)

    num_data = data_df.shape[0]

    column_names = info['column_names'] if info['column_names'] else data_df.columns.tolist()
 
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    idx_mapping, inverse_idx_mapping, idx_name_mapping = get_column_name_mapping(data_df, num_col_idx, cat_col_idx, target_col_idx, column_names)
    print('num_col_idx', num_col_idx)
    print('cat_col_idx', cat_col_idx)
    print('column_names', column_names)
    num_columns = [column_names[i] for i in num_col_idx]
    cat_columns = [column_names[i] for i in cat_col_idx]
    target_columns = [column_names[i] for i in target_col_idx]
    print(num_columns)
    print(cat_columns)
    # print(target_columns)

    if info['test_path']:

        # if testing data is given
        test_path = info['test_path']

        with open(test_path, 'r') as f:
            lines = f.readlines()[1:]
            test_save_path = f'data/{name}/test.data'
            if not os.path.exists(test_save_path):
                with open(test_save_path, 'a') as f1:
                    for line in lines:
                        save_line = line.strip('\n').strip('.')
                        f1.write(f'{save_line}\n')

        test_df = pd.read_csv(test_save_path)
        train_df = data_df

    else:  
        # Train/ Test Split, 90% Training, 10% Testing (Validation set will be selected from Training set)

        num_train = int(num_data*0.9)
        num_test = num_data - num_train

        train_df, test_df, seed = train_val_test_split(data_df, cat_columns, num_train, num_test)
        # print(data_df)
        # print(train_df)

    """ cutmix """
    print(data_df.columns)
    print(train_df.shape)
    print(type(train_df.shape[0]))
    num_new_samples = int(train_df.shape[0]*0.3)
    print(name)
    if name == 'adult':
        label_idx = 14
    elif name == 'default':
        label_idx = 23
    elif name == 'shoppers':
        label_idx = 17
    elif name == 'magic':
        label_idx = 10
        num_new_samples = int(train_df.shape[0] * 2.0)

    subsets = feature_clustering_mixed(train_df, column_names, threshold=0.3)
    print(subsets)
    subsets = [[data_df.columns.get_loc(col) for col in group] for group in subsets]
    print('subsets', subsets)
    train_df = cutmix_tabular_cluster(train_df, label_idx, num_new_samples, subsets)

    print(train_df.shape)
    """ use part of training """
    # train_percent = 0.5
    # train_df = train_df.sample(frac=train_percent, random_state=50)
    # train_percent = 0.6  # 30% data
    # train_df = train_df.sample(frac=train_percent, random_state=50)
    # train_percent = 0.3333333  # 10% data
    # train_df = train_df.sample(frac=train_percent, random_state=25)
    print('train_df.shape', train_df.shape)
    print('test_df.shape', test_df.shape)

    train_df.columns = range(len(train_df.columns))
    test_df.columns = range(len(test_df.columns))

    print(name, train_df.shape, test_df.shape, data_df.shape)
    print(train_df)
    col_info = {}

    for col_idx in num_col_idx:
        col_info[col_idx] = {}
        col_info['type'] = 'numerical'
        col_info['max'] = float(train_df[col_idx].max())
        col_info['min'] = float(train_df[col_idx].min())

    for col_idx in cat_col_idx:
        col_info[col_idx] = {}
        col_info['type'] = 'categorical'
        col_info['categorizes'] = list(set(train_df[col_idx]))
        print(col_info['categorizes'])
        print(col_idx)

    for col_idx in target_col_idx:
        if info['task_type'] == 'regression':
            col_info[col_idx] = {}
            col_info['type'] = 'numerical'
            col_info['max'] = float(train_df[col_idx].max())
            col_info['min'] = float(train_df[col_idx].min())
        else:
            col_info[col_idx] = {}
            col_info['type'] = 'categorical'
            col_info['categorizes'] = list(set(train_df[col_idx]))
            print(col_info['categorizes'])
            print(col_idx)

    info['column_info'] = col_info

    train_df.rename(columns = idx_name_mapping, inplace=True)
    test_df.rename(columns = idx_name_mapping, inplace=True)

    for col in num_columns:
        train_df.loc[train_df[col] == '?', col] = np.nan
    for col in cat_columns:
        train_df.loc[train_df[col] == '?', col] = 'nan'
    for col in num_columns:
        test_df.loc[test_df[col] == '?', col] = np.nan
    for col in cat_columns:
        test_df.loc[test_df[col] == '?', col] = 'nan'

    X_num_train = train_df[num_columns].to_numpy().astype(np.float32)
    X_cat_train = train_df[cat_columns].to_numpy()
    y_train = train_df[target_columns].to_numpy()

    print(test_df[num_columns])
    X_num_test = test_df[num_columns].to_numpy().astype(np.float32)
    X_cat_test = test_df[cat_columns].to_numpy()
    y_test = test_df[target_columns].to_numpy()


    save_dir = f'data/{name}'
    np.save(f'{save_dir}/X_num_train.npy', X_num_train)
    np.save(f'{save_dir}/X_cat_train.npy', X_cat_train)
    np.save(f'{save_dir}/y_train.npy', y_train)

    np.save(f'{save_dir}/X_num_test.npy', X_num_test)
    np.save(f'{save_dir}/X_cat_test.npy', X_cat_test)
    np.save(f'{save_dir}/y_test.npy', y_test)

    train_df[num_columns] = train_df[num_columns].astype(np.float32)
    test_df[num_columns] = test_df[num_columns].astype(np.float32)


    train_df.to_csv(f'{save_dir}/train.csv', index = False)
    test_df.to_csv(f'{save_dir}/test.csv', index = False)

    if not os.path.exists(f'synthetic/{name}'):
        os.makedirs(f'synthetic/{name}')

    train_df.to_csv(f'synthetic/{name}/real.csv', index = False)
    test_df.to_csv(f'synthetic/{name}/test.csv', index = False)

    print('Numerical', X_num_train.shape)
    print('Categorical', X_cat_train.shape)

    info['column_names'] = column_names
    info['train_num'] = train_df.shape[0]
    info['test_num'] = test_df.shape[0]

    info['idx_mapping'] = idx_mapping
    info['inverse_idx_mapping'] = inverse_idx_mapping
    info['idx_name_mapping'] = idx_name_mapping

    metadata = {'columns': {}}
    task_type = info['task_type']
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    for i in num_col_idx:
        metadata['columns'][i] = {}
        metadata['columns'][i]['sdtype'] = 'numerical'
        metadata['columns'][i]['computer_representation'] = 'Float'

    for i in cat_col_idx:
        metadata['columns'][i] = {}
        metadata['columns'][i]['sdtype'] = 'categorical'


    if task_type == 'regression':

        for i in target_col_idx:
            metadata['columns'][i] = {}
            metadata['columns'][i]['sdtype'] = 'numerical'
            metadata['columns'][i]['computer_representation'] = 'Float'

    else:
        for i in target_col_idx:
            metadata['columns'][i] = {}
            metadata['columns'][i]['sdtype'] = 'categorical'

    info['metadata'] = metadata

    with open(f'{save_dir}/info.json', 'w') as file:
        json.dump(info, file, indent=4)

    print(f'Processing and Saving {name} Successfully!')

    print(name)
    print('Total', info['train_num'] + info['test_num'])
    print('Train', info['train_num'])
    print('Test', info['test_num'])
    if info['task_type'] == 'regression':
        num = len(info['num_col_idx'] + info['target_col_idx'])
        cat = len(info['cat_col_idx'])
    else:
        cat = len(info['cat_col_idx'] + info['target_col_idx'])
        num = len(info['num_col_idx'])
    print('Num', num)
    print('Cat', cat)


if __name__ == "__main__":

    if args.dataname:
        process_data(args.dataname)
    else:
        for name in ['adult', 'default', 'shoppers', 'magic']:
        # for name in ['magic']:
            process_data(name)

        

