import pandas as pd
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from sklearn.preprocessing import OneHotEncoder

column_indices = {
        'magic': {
            'numerical': [0,1,2,3,4,5,6,7,8,9],
            'categorical': [10]
        },
        'shoppers': {
            'numerical': [0,1,2,3,4,5,6,7,8,9],
            'categorical': [10,11,12,13,14,15,16,17]
        },
        'adult': {
            'numerical': [0,2,4,10,11,12],
            'categorical': [1,3,5,6,7,8,9,13,14]
        },
        'default': {
            'numerical': [0,4,11,12,13,14,15,16,17,18,19,20,21,22],
            'categorical': [1,2,3,5,6,7,8,9,10,23]
        },
        'Churn_Modelling': {
            'numerical': [0,3,4,5,6,9],
            'categorical': [1,2,7,8,10]
        },
        'cardio_train': {
            'numerical': [0,2,3,4,5],
            'categorical': [1,6,7,8,9,10,11]
        },
        'wilt': {
            'numerical': [1,2,3,4,5],
            'categorical': [0]
        },
        'MiniBooNE': {
            'numerical': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
            'categorical': [0]
        }
    }


def baseline_augmentation(data_df, dataset_name, label_column_idx, num_generate):

    new_samples = pd.DataFrame(columns=data_df.columns)
    # print('new_samples', new_samples)
    if dataset_name not in column_indices:
        raise ValueError(f"Dataset '{dataset_name}' not found in column_indices.")

    numerical_cols = column_indices[dataset_name]['numerical']
    categorical_cols = column_indices[dataset_name]['categorical']

    # Learn distributions for numerical features
    numerical_distributions = {}
    for col in numerical_cols:
        col_data = data_df.iloc[:, col]
        mean, std = col_data.mean(), col_data.std()
        numerical_distributions[col] = (mean, std)

    # Learn frequencies for categorical features
    categorical_frequencies = {}
    for col in categorical_cols:
        freq = data_df.iloc[:, col].value_counts(normalize=True)
        categorical_frequencies[col] = freq

    for _ in range(num_generate):
        new_sample = data_df.iloc[0].copy()  # Initialize new sample

        # Sample numerical features
        for col in numerical_cols:
            mean, std = numerical_distributions[col]
            new_sample.iloc[col] = np.random.normal(mean, std)

        # Sample categorical features
        for col in categorical_cols:
            probabilities = categorical_frequencies[col]
            new_sample.iloc[col] = np.random.choice(probabilities.index, p=probabilities.values)

        # Keep label unchanged by randomly sampling a label from the dataset
        new_sample.iloc[label_column_idx] = data_df.iloc[:, label_column_idx].sample(1).iloc[0]

        # Append the new sample
        new_samples = pd.concat([new_samples, pd.DataFrame([new_sample])], ignore_index=True)

    # Combine the original dataset with the new samples
    augmented_data = pd.concat([data_df, new_samples], ignore_index=True)

    return augmented_data

def generate_syn_sample(data, dataset_name, num_sample=50, num_iterations=100):

    if dataset_name not in column_indices:
        raise ValueError(f"Dataset '{dataset_name}' not found in column_indices.")


    numerical_cols = column_indices[dataset_name]['numerical']
    categorical_cols = column_indices[dataset_name]['categorical']
    all_cols = numerical_cols + categorical_cols

    generated_samples = []

    for _ in range(num_iterations):

        base_samples = data.sample(num_sample)

        for _, base_sample in base_samples.iterrows():
            base_sample = base_sample.copy()


            chosen_col = random.choice(all_cols)


            if chosen_col in numerical_cols:
                base_sample[chosen_col] *= 10
            elif chosen_col in categorical_cols:
                base_sample[chosen_col] = 'unknown'


            generated_samples.append(base_sample)

    generated_df = pd.DataFrame(generated_samples)
    return generated_df

def mixup_tabular(data_df, name, num_generate, alpha=0.4):

    # Validate dataset name and retrieve column indices
    if name not in column_indices:
        raise ValueError(f"Dataset '{name}' is not defined in column_indices.")

    numerical_indices = column_indices[name]['numerical']
    categorical_indices = column_indices[name]['categorical']

    # Combine all feature indices (numerical + categorical)
    all_feature_indices = numerical_indices + categorical_indices

    # Separate numerical and categorical columns
    numerical_columns = data_df.iloc[:, numerical_indices]
    categorical_columns = data_df.iloc[:, categorical_indices]

    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    onehot_encoded = encoder.fit_transform(categorical_columns)
    print(onehot_encoded.shape)
    print(numerical_columns.shape)
    # Combine numerical and one-hot encoded categorical features
    combined_features = np.hstack([numerical_columns.values, onehot_encoded])
    print(combined_features.shape)
    # Initialize result list for mixed samples
    mixed_samples = []

    for _ in range(num_generate):
        # Select two random samples
        # print(combined_features.shape)
        # Ensure indices are selected within the valid range of combined_features
        idx1, idx2 = np.random.choice(combined_features.shape[0], size=2, replace=False)

        sample1 = combined_features[idx1]
        sample2 = combined_features[idx2]

        # Generate mixup coefficient
        lambda_value = np.random.beta(alpha, alpha)

        # Interpolate features
        mixed_features = lambda_value * sample1 + (1 - lambda_value) * sample2
        # print(mixed_features.shape)
        # Decode one-hot back to categorical values for categorical features
        num_features = len(numerical_indices)
        # print(num_features)
        # print(mixed_features[num_features:].reshape(1, -1).shape)
        onehot_decoded = encoder.inverse_transform(mixed_features[num_features:].reshape(1, -1))
        # print(onehot_decoded[0])
        # Combine back to DataFrame format
        mixed_sample = list(mixed_features[:num_features]) + list(onehot_decoded[0])
        mixed_samples.append(mixed_sample)

    # Combine mixed samples with original data
    mixed_samples_df = pd.DataFrame(mixed_samples, columns=list(numerical_columns.columns) + list(categorical_columns.columns))

    # Return the expanded dataset
    return pd.concat([data_df, mixed_samples_df], ignore_index=True)
def smote_tabular(data_df, label_column_idx, name, num_generate, k_neighbors=5):

    # Validate dataset name and retrieve column indices
    if name not in column_indices:
        raise ValueError(f"Dataset '{name}' is not defined in column_indices.")

    numerical_indices = column_indices[name]['numerical']
    categorical_indices = column_indices[name]['categorical']

    # Separate labels and features
    labels = data_df.iloc[:, label_column_idx]
    features = data_df

    # Extract categorical and numerical columns based on indices
    categorical_columns = features.iloc[:, categorical_indices].columns
    numerical_columns = features.iloc[:, numerical_indices].columns


    # Convert categorical columns to category type for consistency
    for col in categorical_columns:
        features[col] = features[col].astype('category')

    # Initialize the result list
    new_samples = []

    # Identify minority class
    label_counts = Counter(labels)
    minority_class = min(label_counts, key=label_counts.get)

    # Filter samples belonging to the minority class
    minority_samples = data_df[data_df.iloc[:, label_column_idx] == minority_class]

    # Use Nearest Neighbors to find similar samples within the minority class
    minority_features = minority_samples.iloc[:, numerical_indices]
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(minority_features)
    neighbors = nn.kneighbors(minority_features, return_distance=False)

    # Generate synthetic samples
    for _ in range(num_generate):
        # Select a random sample from the minority class
        idx = random.choice(range(len(minority_samples)))
        base_sample = minority_samples.iloc[idx]

        # Select a random neighbor
        neighbor_idx = random.choice(neighbors[idx][1:])
        neighbor_sample = minority_samples.iloc[neighbor_idx]

        # Interpolate numerical features
        lambda_value = np.random.uniform(0, 1)
        new_sample_num = (
                lambda_value * base_sample[numerical_columns] +
                (1 - lambda_value) * neighbor_sample[numerical_columns]
        )

        # Handle categorical features
        new_sample_cat = base_sample[categorical_columns].copy()
        for col in categorical_columns:
            if random.random() < 0.5:
                new_sample_cat[col] = neighbor_sample[col]

        # Combine numerical, categorical, and label features
        new_sample = pd.concat([new_sample_num, new_sample_cat])
        new_sample[data_df.columns[label_column_idx]] = minority_class
        new_samples.append(new_sample)

    # Convert new samples to a DataFrame
    new_samples_df = pd.DataFrame(new_samples)

    # Combine the original data with the synthetic samples
    expanded_data = pd.concat([data_df, new_samples_df], ignore_index=True)

    return expanded_data

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

    # print('data_df', data_df)
    num_subsets = len(subset)
    new_samples = pd.DataFrame(columns=data_df.columns)

    for _ in range(num_generate):

        lambda_value = np.random.uniform(0, 1)
        sample1 = data_df.sample(1).iloc[0]


        sample1_label = sample1.iloc[label_column_idx]
        same_label_data = data_df[data_df.iloc[:, label_column_idx] == sample1_label]
        sample2 = same_label_data.sample(1).iloc[0]

        num_subsets_to_mix = int(lambda_value * num_subsets)
        # print('subset', subset)

        mixed_subsets = random.sample(subset, min(num_subsets_to_mix, num_subsets))
        # print('mixed_subsets', mixed_subsets)

        new_sample = sample1.copy()

        switched_features = set()
        for group in mixed_subsets:
            # print('group', group)
            new_sample.iloc[group] = sample2.iloc[group]
            switched_features.update(group)

        # total_features = set(range(data_df.shape[1]))
        # print('total_features', total_features)
        # sample1_features = total_features - switched_features
        # print('sample1_features', sample1_features)
        # sample2_features = switched_features
        # print('sample2_features', sample2_features)

        # if len(sample1_features) > len(sample2_features):
        new_sample.iloc[label_column_idx] = sample1_label
        # else:
        #     new_sample.iloc[label_column_idx] = sample2.iloc[label_column_idx]


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
