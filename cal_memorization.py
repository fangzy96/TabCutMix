import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def custom_distance(X, Y, numerical_cols, categorical_cols):

    distances = np.zeros(X.shape[0])

    if numerical_cols:
        X_num = X[numerical_cols].astype(float).values
        Y_num = Y[numerical_cols].astype(float).values.reshape(1, -1)
        num_diff = X_num - Y_num

        euclidean_distances = np.sqrt(np.sum(num_diff ** 2, axis=1))
        scaler = MinMaxScaler()
        normalized_distances = scaler.fit_transform(euclidean_distances.reshape(-1, 1)).flatten()
        distances += normalized_distances

    if categorical_cols:
        X_cat = X[categorical_cols].values
        Y_cat = Y[categorical_cols].values.reshape(1, -1)
        cat_diff = (X_cat != Y_cat).astype(float)
        cat_distances = cat_diff.sum(axis=1)
        distances += cat_distances

    average_distances = distances / (len(numerical_cols) + len(categorical_cols))

    return average_distances


def cal_memorization(dataname, generated_path, train_data):

    generated_data = pd.read_csv(generated_path)[:1]
    train_data = train_data[:1]

    column_indices = {
        'magic': {
            'numerical': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'categorical': [10]
        },
        'shoppers': {
            'numerical': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'categorical': [10, 11, 12, 13, 14, 15, 16, 17]
        },
        'adult': {
            'numerical': [0, 2, 4, 10, 11, 12],
            'categorical': [1, 3, 5, 6, 7, 8, 9, 13, 14]
        },
        'default': {
            'numerical': [0, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
            'categorical': [1, 2, 3, 5, 6, 7, 8, 9, 10, 23]
        }
    }

    if dataname in column_indices:
        numerical_cols = column_indices[dataname]['numerical']
        categorical_cols = column_indices[dataname]['categorical']
    else:
        print('Invalid dataname.')
        return None

    numerical_col_names = train_data.columns[numerical_cols].tolist()
    categorical_col_names = train_data.columns[categorical_cols].tolist()

    replicate_count = 0


    for index, W in generated_data.iterrows():
        distances = custom_distance(train_data, W, numerical_col_names, categorical_col_names)
        min_index = np.argmin(distances)
        min_distance = distances[min_index]
        distances[min_index] = np.inf
        second_min_index = np.argmin(distances)
        second_min_distance = distances[second_min_index]

        ratio = min_distance / second_min_distance


        if ratio < 1 / 3:
            replicate_count += 1

    replicate_ratio = replicate_count / len(generated_data)
    print(f"{dataname.capitalize()} - Percent of replicate: {replicate_ratio:.2%}")
    return replicate_ratio


def main():
    datasets = ['shoppers']

    for dataset in datasets:
        generated_path = f'synthetic/{dataset}/tabsyn.csv'
        train_data_path = f'synthetic/{dataset}/real_100.csv'
        train_data = pd.read_csv(train_data_path)

        cal_memorization(dataset, generated_path, train_data)


if __name__ == "__main__":
    main()
