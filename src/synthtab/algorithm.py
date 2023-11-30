from copy import deepcopy
from tabddpm import GaussianMultinomialDiffusion
from tabddpm.modules import MLPDiffusion, ResNetDiffusion
import lib
from rich import print
import tomli
import torch
import numpy as np
import pandas as pd

class Trainer:
    def __init__(self, diffusion, train_iter, lr, weight_decay, steps, device=torch.device('cuda')):
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
        self.log_every = 100
        self.print_every = 500
        self.ema_every = 1000

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad()
        loss_multi, loss_gauss = self.diffusion.mixed_loss(x, out_dict)
        loss = loss_multi + loss_gauss
        loss.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss
    
    def update_ema(self, target_params, source_params, rate=0.999):
        """
        Update target parameters to be closer to those of source parameters using
        an exponential moving average.
        :param target_params: the target parameter sequence.
        :param source_params: the source parameter sequence.
        :param rate: the EMA rate (closer to 1 means slower).
        """
        for targ, src in zip(target_params, source_params):
            targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)

    def run_loop(self):
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        curr_count = 0
        while step < self.steps:
            x, out_dict = next(self.train_iter)
            out_dict = {'y': out_dict}
            batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss_multi += batch_loss_multi.item() * len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)

            if (step + 1) % self.log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                if (step + 1) % self.print_every == 0:
                    print(f'Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}')
                self.loss_history.loc[len(self.loss_history)] =[step + 1, mloss, gloss, mloss + gloss]
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0

            self.update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

            step += 1

class Algorithm:
    def __init__(self, config, dataset) -> None:
        self.algo = config['algorithm']
        
        model_type = 'mlp'

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(device)
        batch_size = 4096

        with open('config.toml', 'rb') as f:
            t_c = tomli.load(f)

            print(t_c['model_params'])

            model = self.__get_model__(
                model_type,
                t_c['model_params'],
                n_num_features=config['n_num_features'],
                #This depends on categorical, part is the partition (train,test...), either [] or the list with the categorical features dataset.get_category_sizes('train')
                category_sizes=dataset.get_category_sizes()
            )
            model.to(device)
            #get rid of lib if possible
            train_loader = lib.prepare_fast_dataloader(dataset, split='train', batch_size=batch_size)

            K = np.array(dataset.get_category_sizes())
            if len(K) == 0 or t_c['cat_encoding'] == 'one-hot':
                K = np.array([0])

            self.diffusion = GaussianMultinomialDiffusion(
                num_classes=K,
                num_numerical_features=config['n_num_features'],
                denoise_fn=model,
                gaussian_loss_type='mse',
                num_timesteps=10000,
                scheduler='cosine',
                device=device
            )
            self.diffusion.to(device)
            self.diffusion.train()

            trainer = Trainer(
                self.diffusion,
                train_loader,
                lr=0.002,
                weight_decay=1e-4,
                steps=10000,
                device=device
            )
            trainer.run_loop()

            num_numerical_features_ = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0
            d_in = np.sum(K) + num_numerical_features_
            t_c['model_params']['d_in'] = int(d_in)
            # model = get_model(
            #     model_type,
            #     model_params,
            #     num_numerical_features_,
            #     category_sizes=dataset.get_category_sizes('train')
            # )

            # model.load_state_dict(
            #     torch.load(model_path, map_location="cpu")
            # )

            # diffusion = GaussianMultinomialDiffusion(
            #     K,
            #     num_numerical_features=num_numerical_features_,
            #     denoise_fn=model, num_timesteps=num_timesteps, 
            #     gaussian_loss_type=gaussian_loss_type, scheduler=scheduler, device=device
            # )

            # diffusion.to(device)
            self.diffusion.eval()

            disbalance = 'fix'
            num_samples = 100

            _, empirical_class_dist = torch.unique(torch.from_numpy(dataset.y['train']), return_counts=True)
            # empirical_class_dist = empirical_class_dist.float() + torch.tensor([-5000., 10000.]).float()
            if disbalance == 'fix':
                empirical_class_dist[0], empirical_class_dist[1] = empirical_class_dist[1], empirical_class_dist[0]
                x_gen, y_gen = self.diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=False)

            elif disbalance == 'fill':
                ix_major = empirical_class_dist.argmax().item()
                val_major = empirical_class_dist[ix_major].item()
                x_gen, y_gen = [], []
                for i in range(empirical_class_dist.shape[0]):
                    if i == ix_major:
                        continue
                    distrib = torch.zeros_like(empirical_class_dist)
                    distrib[i] = 1
                    num_samples = val_major - empirical_class_dist[i].item()
                    x_temp, y_temp = self.diffusion.sample_all(num_samples, batch_size, distrib.float(), ddim=False)
                    x_gen.append(x_temp)
                    y_gen.append(y_temp)
                
                x_gen = torch.cat(x_gen, dim=0)
                y_gen = torch.cat(y_gen, dim=0)

            else:
                # Aqui sampleamos
                x_gen, y_gen = self.diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=False)

            X_gen, y_gen = x_gen.numpy(), y_gen.numpy()
            # num_numerical_features = 66
            # num_numerical_features = num_numerical_features + int(dataset.is_regression and not model_params["is_y_cond"])

            # X_num_ = X_gen
            # # if num_numerical_features < X_gen.shape[1]:
            # #     np.save(os.path.join(parent_dir, 'X_cat_unnorm'), X_gen[:, num_numerical_features:])
            # #     # _, _, cat_encoder = lib.cat_encode({'train': X_cat_real}, T_dict['cat_encoding'], y_real, T_dict['seed'], True)
            # #     if T_dict['cat_encoding'] == 'one-hot':
            # #         X_gen[:, num_numerical_features:] = to_good_ohe(dataset.cat_transform.steps[0][1], X_num_[:, num_numerical_features:])
            # #     X_cat = dataset.cat_transform.inverse_transform(X_gen[:, num_numerical_features:])

            # if num_numerical_features_ != 0:
            #     # np.save(os.path.join(parent_dir, 'X_num_unnorm'), X_gen[:, :num_numerical_features])
            #     X_num_ = dataset.num_transform.inverse_transform(X_gen[:, :num_numerical_features])
            #     X_num = X_num_[:, :num_numerical_features]

            #     # X_num_real = np.load(os.path.join(real_data_path, "X_num_train.npy"), allow_pickle=True)
            #     # disc_cols = []
            #     # for col in range(X_num_real.shape[1]):
            #     #     uniq_vals = np.unique(X_num_real[:, col])
            #     #     if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
            #     #         disc_cols.append(col)
            #     # print("Discrete cols:", disc_cols)
            #     if t_c['model_params']['num_classes'] == 0:
            #         y_gen = X_num[:, 0]
            #         X_num = X_num[:, 1:]
                # if len(disc_cols):
                #     X_num = round_columns(X_num_real, X_num, disc_cols)

            print(X_gen, y_gen)
            print(dataset.X_num)

    def __get_model__(self,
        model_name,
        model_params,
        n_num_features,
        category_sizes
    ): 
        print(model_name)
        if model_name == 'mlp':
            model = MLPDiffusion(**model_params)
        elif model_name == 'resnet':
            model = ResNetDiffusion(**model_params)
        else:
            raise "Unknown model!"
        return model

    def __str__(self) -> str:
        return self.algo