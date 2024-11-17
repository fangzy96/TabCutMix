import pandas as pd
import numpy as np
import random


def cutmix_tabular(data_df, label_column_idx, num_generate):


    feature_columns_idx = [i for i in range(data_df.shape[1]) if i != label_column_idx]
    new_samples = pd.DataFrame(columns=data_df.columns)

    for _ in range(num_generate):

        lambda_value = np.random.uniform(0, 1)
        sample1 = data_df.sample(1).iloc[0]


        sample1_label = sample1.iloc[label_column_idx]
        same_label_data = data_df[data_df.iloc[:, label_column_idx] == sample1_label]
        sample2 = same_label_data.sample(1).iloc[0]


        num_features_to_mix = int(lambda_value * len(feature_columns_idx))
        mixed_features_idx = random.sample(feature_columns_idx, num_features_to_mix)


        new_sample = sample1.copy()
        new_sample.iloc[mixed_features_idx] = sample2.iloc[mixed_features_idx]

        new_sample.iloc[label_column_idx] = sample1_label


        new_samples = pd.concat([new_samples, pd.DataFrame([new_sample])], ignore_index=True)


    expanded_data = pd.concat([data_df, new_samples], ignore_index=True)

    return expanded_data


def cutmix_tabular_cluster(data_df, label_column_idx, num_generate, subset):
    """
    在输入数据上应用 CutMix 方法，结合特征聚类。

    参数:
    - data_df: pandas DataFrame, 原始数据。
    - label_column_idx: int, 标签列的索引。
    - num_generate: int, 要生成的样本数量。
    - subset: List[List[int]], 特征聚类结果，每个子列表表示一个特征集合。

    返回:
    - expanded_data: pandas DataFrame, 原始数据和新生成样本的组合。
    """
    # 获取子集数量
    # print('data_df', data_df)
    num_subsets = len(subset)
    new_samples = pd.DataFrame(columns=data_df.columns)

    for _ in range(num_generate):
        # 随机生成 lambda 值
        lambda_value = np.random.uniform(0, 1)
        # print(lambda_value)
        # 随机选择第一个样本
        sample1 = data_df.sample(1).iloc[0]

        # 根据第一个样本的标签找到相同标签的样本作为第二个样本
        sample1_label = sample1.iloc[label_column_idx]
        same_label_data = data_df[data_df.iloc[:, label_column_idx] == sample1_label]
        sample2 = same_label_data.sample(1).iloc[0]

        # 计算需要混合的子集数量（基于子集的数量，而非特征总数）
        num_subsets_to_mix = int(lambda_value * num_subsets)
        # print('subset', subset)
        # print('num_subsets', num_subsets)
        # print('num_subsets_to_mix', num_subsets_to_mix)

        # 随机选择要混合的子集
        mixed_subsets = random.sample(subset, min(num_subsets_to_mix, num_subsets))
        # print('mixed_subsets', mixed_subsets)

        # 创建新样本
        new_sample = sample1.copy()

        # 混合特征：处理被选中的子集
        switched_features = set()
        for group in mixed_subsets:
            # print('group', group)
            new_sample.iloc[group] = sample2.iloc[group]
            switched_features.update(group)

        # 检查特征占比
        # total_features = set(range(data_df.shape[1]))
        # print('total_features', total_features)
        # sample1_features = total_features - switched_features
        # print('sample1_features', sample1_features)
        # sample2_features = switched_features
        # print('sample2_features', sample2_features)

        # 决定标签值
        # if len(sample1_features) > len(sample2_features):
        new_sample.iloc[label_column_idx] = sample1_label
        # else:
        #     new_sample.iloc[label_column_idx] = sample2.iloc[label_column_idx]

        # 添加到新样本集中
        new_samples = pd.concat([new_samples, pd.DataFrame([new_sample])], ignore_index=True)

    # 合并原始数据和新生成的数据
    expanded_data = pd.concat([data_df, new_samples], ignore_index=True)

    return expanded_data

if __name__ == '__main__':

    data = {
        'Feature1': [1, 2, 3, 4, 5],
        'Feature2': [10, 20, 30, 40, 50],
        'Feature3': [100, 200, 300, 400, 500],
        'Label': [0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)

    expanded_df = cutmix_tabular(df, label_column_idx=3, num_generate=5)

    print(expanded_df)
