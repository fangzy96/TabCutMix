import json
import os
import random
import numpy as np
import pandas as pd

def update_indices(original_indices, remaining_indices):

    mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(remaining_indices))}
    return [mapping[idx] for idx in original_indices if idx in mapping]


def modify_adult_json(input_path, output_path, cutoff, seed=19):

    random.seed(seed)

    with open(input_path, 'r') as f:
        data = json.load(f)

    num_col_idx = data['num_col_idx']
    cat_col_idx = data['cat_col_idx']
    target_col_idx = data['target_col_idx']
    print(f'num_col_idx: {num_col_idx}')
    print(f'cat_col_idx: {cat_col_idx}')
    print(f'target_col_idx: {target_col_idx}')

    all_col_idx = num_col_idx + cat_col_idx

    num_to_keep = int(len(num_col_idx) * cutoff)
    cat_to_keep = int(len(cat_col_idx) * cutoff)

    remaining_num_cols = sorted(random.sample(num_col_idx, num_to_keep))
    remaining_cat_cols = sorted(random.sample(cat_col_idx, cat_to_keep))

    remaining_cols = sorted(remaining_num_cols + remaining_cat_cols)

    data['column_names'] = [data['column_names'][i] for i in remaining_cols + data['target_col_idx']]
    data['column_info'] = {data['column_names'][i]: data['column_info'][data['column_names'][i]] for i in
                           range(len(data['column_names']))}

    data['num_col_idx'] = update_indices(num_col_idx, remaining_cols)
    data['cat_col_idx'] = update_indices(cat_col_idx, remaining_cols)

    data['target_col_idx'] = [len(data['column_names']) - 1]

    csv_path = data['data_path']
    df = pd.read_excel(csv_path) if csv_path.endswith('.xls') else pd.read_csv(csv_path)

    remaining_feature_names = [df.columns[i] for i in remaining_cols + target_col_idx]
    df = df[remaining_feature_names]

    cutoff_str = str(cutoff)
    new_csv_path = f"{os.path.splitext(csv_path)[0]}_{cutoff_str}.csv"

    df.to_csv(new_csv_path, index=False)

    data['data_path'] = new_csv_path
    # if testing data is given
    csv_path = data['test_path']
    # from 100% to 70%
    with open(csv_path, 'r') as f:
        lines = f.readlines()[1:]
        test_save_path = f'data/adult/test.data'
        if not os.path.exists(test_save_path):
            with open(test_save_path, 'a') as f1:
                for line in lines:
                    save_line = line.strip('\n').strip('.')
                    f1.write(f'{save_line}\n')
    print(test_save_path)
    df = pd.read_csv(test_save_path, header=None)
    # from 70% to others
    # df = pd.read_csv(csv_path, header=None)
    print(df)

    remaining_feature_names = [df.columns[i] for i in remaining_cols + target_col_idx]
    df = df[remaining_feature_names]
    data['test_path'] = "data/adult/test.csv"

    df.to_csv(data['test_path'], header=False, index=False)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Modified adult JSON saved to {output_path}")

def modify_default_json(input_path, output_path, cutoff, seed=19):
    random.seed(seed)

    with open(input_path, 'r') as f:
        data = json.load(f)

    num_col_idx = data['num_col_idx']
    cat_col_idx = data['cat_col_idx']
    target_col_idx = data['target_col_idx']
    print(f'num_col_idx: {num_col_idx}')
    print(f'cat_col_idx: {cat_col_idx}')
    print(f'target_col_idx: {target_col_idx}')
    num_to_keep = int(len(num_col_idx) * cutoff)
    cat_to_keep = int(len(cat_col_idx) * cutoff)

    remaining_num_cols = sorted(random.sample(num_col_idx, num_to_keep))
    remaining_cat_cols = sorted(random.sample(cat_col_idx, cat_to_keep))
    print('remaining_num_cols', remaining_num_cols)
    print('remaining_cat_cols', remaining_cat_cols)

    remaining_cols = sorted(remaining_num_cols + remaining_cat_cols)
    print('remaining_cols', remaining_cols)

    data['num_col_idx'] = update_indices(num_col_idx, remaining_cols)
    data['cat_col_idx'] = update_indices(cat_col_idx, remaining_cols)
    print('remaining_num_cols', data['num_col_idx'])
    print('remaining_cat_cols', data['cat_col_idx'])

    data['target_col_idx'] = [len(remaining_cols)]
    print('target_col_idx', data['target_col_idx'])

    csv_path = data['data_path']
    df = pd.read_excel(csv_path) if csv_path.endswith('.xls') else pd.read_csv(csv_path)

    remaining_feature_names = [df.columns[i] for i in remaining_cols + target_col_idx]
    df = df[remaining_feature_names]

    cutoff_str = str(cutoff)
    new_csv_path = f"{os.path.splitext(csv_path)[0]}_{cutoff_str}.csv"

    df.to_csv(new_csv_path, index=False)

    data['data_path'] = new_csv_path
    print(data)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Modified default JSON saved to {output_path}")


def modify_shoppers_json(input_path, output_path, cutoff, seed=42):
    random.seed(seed)

    with open(input_path, 'r') as f:
        data = json.load(f)

    num_col_idx = data['num_col_idx']
    cat_col_idx = data['cat_col_idx']
    target_col_idx = data['target_col_idx']
    print(f'num_col_idx: {num_col_idx}')
    print(f'cat_col_idx: {cat_col_idx}')
    print(f'target_col_idx: {target_col_idx}')

    num_to_keep = int(len(num_col_idx) * cutoff)
    cat_to_keep = int(len(cat_col_idx) * cutoff)

    remaining_num_cols = sorted(random.sample(num_col_idx, num_to_keep))
    remaining_cat_cols = sorted(random.sample(cat_col_idx, cat_to_keep))
    remaining_cols = sorted(remaining_num_cols + remaining_cat_cols)

    data['num_col_idx'] = update_indices(num_col_idx, remaining_cols)
    data['cat_col_idx'] = update_indices(cat_col_idx, remaining_cols)

    data['target_col_idx'] = [len(remaining_cols)]

    csv_path = data['data_path']
    df = pd.read_excel(csv_path) if csv_path.endswith('.xls') else pd.read_csv(csv_path)

    remaining_feature_names = [df.columns[i] for i in remaining_cols + target_col_idx]
    df = df[remaining_feature_names]
    print(df.columns)

    cutoff_str = str(cutoff)
    new_csv_path = f"{os.path.splitext(csv_path)[0]}_{cutoff_str}.csv"

    df.to_csv(new_csv_path, index=False)

    data['data_path'] = new_csv_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Modified shoppers JSON saved to {output_path}")

def modify_magic_json(input_path, output_path, cutoff, seed=42):
    random.seed(seed)

    with open(input_path, 'r') as f:
        data = json.load(f)

    num_col_idx = data['num_col_idx']
    target_col_idx = data['target_col_idx']
    print(f'num_col_idx: {num_col_idx}')
    print(f'target_col_idx: {target_col_idx}')

    num_to_keep = int(len(num_col_idx) * cutoff)

    remaining_num_cols = sorted(random.sample(num_col_idx, num_to_keep))

    data['num_col_idx'] = update_indices(num_col_idx, remaining_num_cols)

    data['target_col_idx'] = [len(remaining_num_cols)]

    csv_path = data['data_path']
    df = pd.read_excel(csv_path) if csv_path.endswith('.xls') else pd.read_csv(csv_path)

    remaining_feature_names = [df.columns[i] for i in remaining_num_cols + target_col_idx]
    df = df[remaining_feature_names]
    print(df.columns)

    cutoff_str = str(cutoff)
    new_csv_path = f"{os.path.splitext(csv_path)[0]}_{cutoff_str}.csv"

    df.to_csv(new_csv_path, index=False)

    data['data_path'] = new_csv_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Modified magic JSON saved to {output_path}")

def modify_beijing_json(input_path, output_path, cutoff, seed=42):
    random.seed(seed)

    with open(input_path, 'r') as f:
        data = json.load(f)

    num_col_idx = data['num_col_idx']
    cat_col_idx = data['cat_col_idx']
    target_col_idx = data['target_col_idx']
    print(f'num_col_idx: {num_col_idx}')
    print(f'cat_col_idx: {cat_col_idx}')
    print(f'target_col_idx: {target_col_idx}')

    num_to_keep = int(len(num_col_idx) * cutoff)
    cat_to_keep = int(len(cat_col_idx) * cutoff)

    remaining_num_cols = sorted(random.sample(num_col_idx, num_to_keep))
    remaining_cat_cols = sorted(random.sample(cat_col_idx, cat_to_keep))
    remaining_cols = sorted(remaining_num_cols + remaining_cat_cols)

    data['num_col_idx'] = update_indices(num_col_idx, remaining_cols)
    data['cat_col_idx'] = update_indices(cat_col_idx, remaining_cols)

    data['target_col_idx'] = [len(remaining_cols)]

    csv_path = data['data_path']
    df = pd.read_excel(csv_path) if csv_path.endswith('.xls') else pd.read_csv(csv_path)

    remaining_feature_names = [df.columns[i] for i in remaining_cols + target_col_idx]
    df = df[remaining_feature_names]

    cutoff_str = str(cutoff)
    new_csv_path = f"{os.path.splitext(csv_path)[0]}_{cutoff_str}.csv"

    df.to_csv(new_csv_path, index=False)

    data['data_path'] = new_csv_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Modified beijing JSON saved to {output_path}")


cutoff = 0.7
cutoff_str = str(cutoff)

adult_input_json_path = 'data/Info/adult.json'
adult_output_json_path = f'data/Info_{cutoff_str}/adult.json'

default_input_json_path = 'data/Info/default.json'
default_output_json_path = f'data/Info_{cutoff_str}/default.json'

shoppers_input_json_path = 'data/Info/shoppers.json'
shoppers_output_json_path = f'data/Info_{cutoff_str}/shoppers.json'

magic_input_json_path = 'data/Info/magic.json'
magic_output_json_path = f'data/Info_{cutoff_str}/magic.json'

beijing_input_json_path = 'data/Info/beijing.json'
beijing_output_json_path = f'data/Info_{cutoff_str}/beijing.json'

modify_adult_json(adult_input_json_path, adult_output_json_path, cutoff)
# modify_default_json(default_input_json_path, default_output_json_path, cutoff)
# modify_shoppers_json(shoppers_input_json_path, shoppers_output_json_path, cutoff)
# modify_magic_json(magic_input_json_path, magic_output_json_path, cutoff)

