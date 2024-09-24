import pandas as pd
import numpy as np
import random


def cutmix_tabular(data_df, label_column_idx, num_generate):
    """
    对Tabular数据执行Intra-class CutMix操作。

    参数：
        data_df: 原始训练集DataFrame。
        label_column_idx: 标签列的位置索引。
        num_generate: 要生成的新样本数量。

    返回：
        一个扩展后的DataFrame，包含原始和生成的新样本。
    """
    # 获取所有feature列的索引，排除标签列
    feature_columns_idx = [i for i in range(data_df.shape[1]) if i != label_column_idx]

    # 初始化新的DataFrame用于存储生成的新样本
    new_samples = pd.DataFrame(columns=data_df.columns)

    for _ in range(num_generate):
        # 从0到1之间随机抽取lambda值
        lambda_value = np.random.uniform(0, 1)

        # 随机从数据集中选择一个样本
        sample1 = data_df.sample(1).iloc[0]

        # 获取 sample1 的标签
        sample1_label = sample1.iloc[label_column_idx]

        # 从相同标签的数据集中选择第二个样本
        same_label_data = data_df[data_df.iloc[:, label_column_idx] == sample1_label]
        sample2 = same_label_data.sample(1).iloc[0]

        # 计算需要混合的feature的数量（根据lambda决定）
        num_features_to_mix = int(lambda_value * len(feature_columns_idx))
        mixed_features_idx = random.sample(feature_columns_idx, num_features_to_mix)

        # 创建新的样本，将混合的feature替换为sample2的值
        new_sample = sample1.copy()
        new_sample.iloc[mixed_features_idx] = sample2.iloc[mixed_features_idx]

        # 直接使用相同的标签
        new_sample.iloc[label_column_idx] = sample1_label

        # 将新样本添加到生成的新样本DataFrame中
        new_samples = pd.concat([new_samples, pd.DataFrame([new_sample])], ignore_index=True)

    # 将原始数据集和生成的新样本合并
    expanded_data = pd.concat([data_df, new_samples], ignore_index=True)

    return expanded_data


if __name__ == '__main__':
    # 创建一个示例数据集
    data = {
        'Feature1': [1, 2, 3, 4, 5],
        'Feature2': [10, 20, 30, 40, 50],
        'Feature3': [100, 200, 300, 400, 500],
        'Label': [0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)

    # 使用cutmix_tabular生成新样本
    expanded_df = cutmix_tabular(df, label_column_idx=3, num_generate=5)

    # 打印生成的扩展数据集
    print(expanded_df)
