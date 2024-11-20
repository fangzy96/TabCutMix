import os
import sys
import time
import torch
import numpy as np
import pandas as pd

from copy import deepcopy

import src
from utils_train import make_dataset, update_ema
from baselines.tabddpm.models.modules import MLPDiffusion
from baselines.tabddpm.models.gaussian_multinomial_distribution import GaussianMultinomialDiffusion
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import json

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


def cal_memorization(dataname, generated_path, train_data_path):

    generated_data = pd.read_csv(generated_path)
    # train_data_path = f'data/{dataname}/train_100.csv'
    train_data = pd.read_csv(train_data_path)

    column_indices = {
        'magic': {
            'numerical': [0,1,2,3,4,5,6,7,8,9],
            'categorical': [10]
        },
        'beijing': {
            'numerical': [4,5,6,7,9,10,11],
            'categorical': [0,1,2,3,8]
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

        # print(train_data.shape)
        # print(W.shape)
        distances = custom_distance(train_data, W, numerical_col_names, categorical_col_names)
        # print(distances)
        # print(distances.shape)

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

@torch.no_grad()
def split_num_cat_target(syn_data, info, num_inverse, cat_inverse):
    task_type = info['task_type']

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    n_num_feat = len(num_col_idx)
    n_cat_feat = len(cat_col_idx)

    if task_type == 'regression':
        n_num_feat += len(target_col_idx)
    else:
        n_cat_feat += len(target_col_idx)

    syn_num = syn_data[:, :n_num_feat]
    syn_cat = syn_data[:, n_num_feat:]

    syn_num = num_inverse(syn_num).astype(np.float32)
    syn_cat = cat_inverse(syn_cat)

    if info['task_type'] == 'regression':
        syn_target = syn_num[:, :len(target_col_idx)]
        syn_num = syn_num[:, len(target_col_idx):]

    else:
        print(syn_cat.shape)
        syn_target = syn_cat[:, :len(target_col_idx)]
        syn_cat = syn_cat[:, len(target_col_idx):]

    return syn_num, syn_cat, syn_target


def recover_data(syn_num, syn_cat, syn_target, info):
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    idx_mapping = info['idx_mapping']
    idx_mapping = {int(key): value for key, value in idx_mapping.items()}

    syn_df = pd.DataFrame()

    if info['task_type'] == 'regression':
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]]
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]


    else:
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]]
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]

    return syn_df


def sample(
        model_save_path,
        sample_save_path,
        real_data_path,
        batch_size=2000,
        num_samples=0,
        task_type='binclass',
        model_type='mlp',
        model_params=None,
        num_timesteps=1000,
        gaussian_loss_type='mse',
        scheduler='cosine',
        T_dict=None,
        num_numerical_features=0,
        disbalance=None,
        device=torch.device('cuda:0'),
        change_val=False,
        ddim=False,
        steps=1000,
):
    T = src.Transformations(**T_dict)

    D = make_dataset(
        real_data_path,
        T,
        task_type=task_type,
        change_val=False,
    )

    K = np.array(D.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])

    num_numerical_features_ = D.X_num['train'].shape[1] if D.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features_
    model_params['d_in'] = int(d_in)
    model = get_model(
        model_type,
        model_params,
        num_numerical_features_,
        category_sizes=D.get_category_sizes('train')
    )

    model_path = f'{model_save_path}/model.pt'

    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )

    diffusion = GaussianMultinomialDiffusion(
        K,
        num_numerical_features=num_numerical_features_,
        denoise_fn=model, num_timesteps=num_timesteps,
        gaussian_loss_type=gaussian_loss_type, scheduler=scheduler, device=device
    )

    diffusion.to(device)
    diffusion.eval()

    start_time = time.time()
    if not ddim:
        x_gen = diffusion.sample_all(num_samples, batch_size, ddim=False)
    else:
        x_gen = diffusion.sample_all(num_samples, batch_size, ddim=True, steps=steps)

    print('Shape', x_gen.shape)

    syn_data = x_gen
    num_inverse = D.num_transform.inverse_transform
    cat_inverse = D.cat_transform.inverse_transform

    info_path = f'{real_data_path}/info.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse)
    syn_df = recover_data(syn_num, syn_cat, syn_target, info)

    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

    syn_df.rename(columns=idx_name_mapping, inplace=True)
    end_time = time.time()

    print('Sampling time:', end_time - start_time)

    save_path = sample_save_path
    syn_df.to_csv(save_path, index=False)


def sample_main(args, num_samples):
    dataname = args.dataname
    device = f'cuda:{args.gpu}'

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = f'{curr_dir}/configs/{dataname}.toml'
    model_save_path = f'{curr_dir}/ckpt/{dataname}'
    real_data_path = f'data/{dataname}'
    sample_save_path = args.save_path

    args.train = True

    raw_config = src.load_config(config_path)

    ''' 
    Modification of configs
    '''
    print('START SAMPLING')

    sample(
        num_samples=num_samples,
        batch_size=raw_config['sample']['batch_size'],
        disbalance=raw_config['sample'].get('disbalance', None),
        **raw_config['diffusion_params'],
        model_save_path=model_save_path,
        sample_save_path=sample_save_path,
        real_data_path=real_data_path,
        task_type=raw_config['task_type'],
        model_type=raw_config['model_type'],
        model_params=raw_config['model_params'],
        T_dict=raw_config['train']['T'],
        num_numerical_features=raw_config['num_numerical_features'],
        device=device,
        ddim=args.ddim,
        steps=args.steps
    )

def get_model(
    model_name,
    model_params,
    n_num_features,
    category_sizes
): 
    print(model_name)
    if model_name == 'mlp':
        model = MLPDiffusion(**model_params)
    else:
        raise "Unknown model!"
    return model

class Trainer:
    def __init__(self, diffusion, train_iter, lr, weight_decay, steps, model_save_path, device=torch.device('cuda:1'), args=None):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.train_iter = train_iter
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step', 'mloss', 'gloss', 'loss'])
        self.model_save_path = model_save_path
        self.args = args
        columns = list(np.arange(5)*200)
        columns[0] = 1
        columns = ['step'] + columns
 

        self.log_every = 50
        self.print_every = 1
        self.ema_every = 1000

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x):
        x = x.to(self.device)

        self.optimizer.zero_grad()

        loss_multi, loss_gauss = self.diffusion.mixed_loss(x)

        loss = loss_multi + loss_gauss
        loss.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss

    def run_loop(self):
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        curr_count = 0
        self.print_every = 1
        self.log_every = 1

        best_loss = np.inf
        dataname = self.args.dataname
        if dataname == 'default':
            self.steps = 15000
        else:
            self.steps = 100000
        print('Steps: ', self.steps)
        cal_memo_real_data_path = f'synthetic/{dataname}/real.csv'
        sample_save_path = self.args.save_path
        print(cal_memo_real_data_path)
        print(sample_save_path)
        print(self.args)

        replicate_ratio_list, epoch_list = [], []
        batch_size = 1024
        real_df = pd.read_csv(cal_memo_real_data_path)
        num_samples = real_df.shape[0]
        need_steps = (num_samples // batch_size)*10

        print(f'num_samples: {num_samples}')
        epoch = 0
        while step < self.steps:
            start_time = time.time()
            x = next(self.train_iter)[0]
            
            batch_loss_multi, batch_loss_gauss = self._run_step(x)

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss_multi += batch_loss_multi.item() * len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)

            
            if (step + 1) % self.log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                if np.isnan(gloss):
                    print('Finding Nan')
                    break
                
                if (step + 1) % self.print_every == 0:
                    print(f'Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}')
                self.loss_history.loc[len(self.loss_history)] =[step + 1, mloss, gloss, mloss + gloss]

                np.set_printoptions(suppress=True)
          
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0

                if mloss + gloss < best_loss:
                    best_loss = mloss + gloss
                    torch.save(self.diffusion._denoise_fn.state_dict(), os.path.join(self.model_save_path, 'model.pt'))
  
                if (step + 1) % 10000 == 0:
                    torch.save(self.diffusion._denoise_fn.state_dict(), os.path.join(self.model_save_path, f'model_{step+1}.pt'))

                # epoch = ((step + 1) * batch_size) // num_samples

                # if (step + 1) % need_steps == 0:
                #     print(f'Epoch: {epoch}')
                #     sample_main(self.args)  # sample data
                #     cur_replicate_ratio = cal_memorization(dataname, sample_save_path, real_data_path)
                #     replicate_ratio_list.append(cur_replicate_ratio)
                #     epoch_list.append(epoch)
                #     cur_data = {
                #         'Epoch': epoch_list,
                #         'Replicate Ratio': replicate_ratio_list
                #     }
                #

                #     df = pd.DataFrame(cur_data)
                #     epoch += 10

                #     memo_save_path = f'sample/{dataname}/TabDDPM_{dataname}_ratio.csv'
                #     df.to_csv(memo_save_path, index=False)
                #     print(f'Saved replicate ratios and epochs to {memo_save_path}')

            # update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

            step += 1
        sample_main(self.args, num_samples)  # sample data
        cur_replicate_ratio = cal_memorization(dataname, sample_save_path, cal_memo_real_data_path)
        print('cur_replicate_ratio', cur_replicate_ratio)
            # end_time = time.time()
            # print('Time: ', end_time - start_time)

def train(
    model_save_path,
    real_data_path,
    steps = 1000,
    lr = 0.002,
    weight_decay = 1e-4,
    batch_size = 1024,
    task_type = 'binclass',
    model_type = 'mlp',
    model_params = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    T_dict = None,
    num_numerical_features = 0,
    device = torch.device('cuda:0'),
    seed = 0,
    change_val = False,
    args = None
):
    real_data_path = os.path.normpath(real_data_path)

    # zero.improve_reproducibility(seed)

    T = src.Transformations(**T_dict)


    dataset = make_dataset(
        real_data_path,
        T,
        task_type = task_type,
        change_val = False,
    )

    K = np.array(dataset.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])

    num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features
    model_params['d_in'] = d_in
    print(d_in)
    
    print(model_params)
    model = get_model(
        model_type,
        model_params,
        num_numerical_features,
        category_sizes=dataset.get_category_sizes('train')
    )
    model.to(device)

    print(model)

    train_loader = src.prepare_fast_dataloader(dataset, split='train', batch_size=batch_size)

    diffusion = GaussianMultinomialDiffusion(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        device=device
    )

    num_params = sum(p.numel() for p in diffusion.parameters())
    print("the number of parameters", num_params)
    

    diffusion.to(device)

    diffusion.train()

    trainer = Trainer(
        diffusion,
        train_loader,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        model_save_path=model_save_path,
        device=device,
        args=args
    )
    trainer.run_loop()

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    torch.save(diffusion._denoise_fn.state_dict(), os.path.join(model_save_path, 'model.pt'))
    torch.save(trainer.ema_model.state_dict(), os.path.join(model_save_path, 'model_ema.pt'))

    trainer.loss_history.to_csv(os.path.join(model_save_path, 'loss.csv'), index=False)