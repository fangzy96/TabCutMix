import os
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings
import time
import copy
import numpy as np
import pandas as pd
from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import get_input_generate, recover_data, split_num_cat_target
from tabsyn.diffusion_utils import sample
from tqdm import tqdm
from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import get_input_train
from sklearn.preprocessing import StandardScaler, MinMaxScaler
warnings.filterwarnings('ignore')

def create_sample_args(train_args):

    sample_args = copy.deepcopy(train_args)

    sample_args.mode = 'sample'

    sample_args.save_path = f'sample/{train_args.dataname}/{train_args.dataname}_generated.csv'

    return sample_args

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

    print(generated_path)

    generated_data = pd.read_csv(generated_path)
    # train_data_path = f'data/{dataname}/train_100.csv'
    # train_data = pd.read_csv(train_data_path)


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

    if dataname in column_indices:
        numerical_cols = column_indices[dataname]['numerical']
        categorical_cols = column_indices[dataname]['categorical']
    else:
        print('Invalid dataname.')
    print(numerical_cols)
    print(categorical_cols)
    numerical_col_names = train_data.columns[numerical_cols].tolist()
    categorical_col_names = train_data.columns[categorical_cols].tolist()
    print(numerical_col_names)
    print(categorical_col_names)

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
    print(f"Percent of replicate: {replicate_ratio:.2%}")
    return replicate_ratio

def generate_sample(args, model, epoch, num_samples):
    dataname = args.dataname
    device = args.device
    steps = args.steps


    save_path = f"sample/{dataname}/{dataname}_generated_{epoch}.csv"

    train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)
    in_dim = train_z.shape[1]

    mean = train_z.mean(0)

    '''
        Generating samples    
    '''
    start_time = time.time()

    # num_samples = train_z.shape[0]
    sample_dim = in_dim

    x_next = sample(model.denoise_fn_D, num_samples, sample_dim)
    x_next = x_next * 2 + mean.to(device)

    syn_data = x_next.float().cpu().numpy()
    syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse, args.device)

    syn_df = recover_data(syn_num, syn_cat, syn_target, info)

    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

    syn_df.rename(columns=idx_name_mapping, inplace=True)
    print(save_path)
    syn_df.to_csv(save_path, index=False)

    end_time = time.time()
    print('Time:', end_time - start_time)
    print(f'Saving sampled data to {save_path}')
    return dataname, save_path



def main(args):
    sample_args = create_sample_args(args)
    device = args.device
    dataname = args.dataname
    train_100_path = f'synthetic/{dataname}/real.csv'
    train_data_100 = pd.read_csv(train_100_path)
    num_generate = train_data_100.shape[0]

    train_z, _, _, ckpt_path, _ = get_input_train(args)

    print(ckpt_path)

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    in_dim = train_z.shape[1]

    mean, std = train_z.mean(0), train_z.std(0)

    train_z = (train_z - mean) / 2
    train_data = train_z

    batch_size = 4096
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    num_epochs = 10000 + 1

    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    print(denoise_fn)

    num_params = sum(p.numel() for p in denoise_fn.parameters())
    print("the number of parameters", num_params)

    model = Model(denoise_fn=denoise_fn, hid_dim=train_z.shape[1]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20, verbose=True)

    model.train()

    best_loss = float('inf')
    patience = 0
    start_time = time.time()
    replicate_ratio_list, epoch_list = [], []
    for epoch in range(num_epochs):

        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}")

        batch_loss = 0.0
        len_input = 0
        for batch in pbar:
            inputs = batch.float().to(device)
            loss = model(inputs)

            loss = loss.mean()

            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Loss": loss.item()})

        curr_loss = batch_loss / len_input
        scheduler.step(curr_loss)

        if curr_loss < best_loss:
            best_loss = curr_loss
            patience = 0
            torch.save(model.state_dict(), f'{ckpt_path}/model.pt')
        else:
            patience += 1
            if patience == 2000:
                print('Early stopping')
                break

        # if epoch % 10 == 0:
        # torch.save(model.state_dict(), f'{ckpt_path}/model_{epoch}.pt')
    cur_dataname, cur_save_path = generate_sample(sample_args, model, epoch, num_generate)
    # test memorization
    cur_replicate_ratio = cal_memorization(cur_dataname, cur_save_path, train_data_100)
    replicate_ratio_list.append(cur_replicate_ratio)
    epoch_list.append(epoch)
    cur_data = {
        'Epoch': epoch_list,
        'Replicate Ratio': replicate_ratio_list
    }

    df = pd.DataFrame(cur_data)

    memo_save_path = f'sample/{cur_dataname}/{cur_dataname}_ratio.csv'
    df.to_csv(memo_save_path, index=False)
    print(f'Saved replicate ratios and epochs to {memo_save_path}')

    end_time = time.time()
    print('Time: ', end_time - start_time)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training of TabSyn')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'