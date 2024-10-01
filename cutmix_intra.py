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
