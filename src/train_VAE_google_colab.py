"""
A train script that is able to resume training very easily
"""

import os
import urllib.request
import zipfile

import torch.optim
import wget
import sys

from matplotlib import pyplot as plt
from tqdm import tqdm
import yaml
from data import MuseDB_HQ
from torch.utils.data import DataLoader
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt
import librosa
import librosa.display
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


sys.path.insert(0, os.path.join(sys.path[0], './taming-transformers'))
sys.path.insert(0, './taming-transformers')
from taming.models.vqgan import VQModel


class DownloadProgressBar(tqdm):
    def update_to(self, current, total, width=80):
        if self.total is None:
            self.total = total
        self.update(100 * current / total)


def main():
    # change relativ working directory
    google_colab_relativ_path = 'Unknown-Signal-Decomposition'
    os.chdir(os.path.join(os.getcwd(), google_colab_relativ_path, 'src'))

    # download the dataset
    if not os.path.exists('./dataset/MUSDBhq'):
        print('downloading the data...')
        dataset_url = 'https://zenodo.org/record/3338373/files/musdb18hq.zip?download=1'
        # urllib.request.urlretrieve(dataset_url, "musdb18hq.zip")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc='[Downloading zip]', total=None) as t:
            wget.download(url=dataset_url, out="musdb18hq.zip", bar=t.update_to)

        # unzip
        print('unziping...')
        with zipfile.ZipFile(r"musdb18hq.zip", 'r') as zip_ref:
            for member in tqdm(zip_ref.infolist(), desc='[Unzipping]'):
                try:
                    zip_ref.extract(member, './dataset/MUSDBhq')
                except zipfile.error as e:
                    pass

    if not os.path.exists('taming-transformers'):
        # get the taming-transformers git repo
        os.system('git submodule init')

    # read the config file
    with open('config/vae_config.yaml', 'r') as file:
        config = yaml.load(file.read(), Loader=Loader)

    device = 'cuda'

    # get the data
    dataset_train = MuseDB_HQ(**config['data']['train'])
    dataset_test = MuseDB_HQ(**config['data']['test'])

    train_data = DataLoader(dataset=dataset_train, shuffle=True, batch_size=config['data']['batch_size'], num_workers=config['data']['num_workers'])
    test_data = DataLoader(dataset=dataset_test, shuffle=False, batch_size=config['data']['batch_size'], num_workers=config['data']['num_workers'])

    # load the model
    vae: nn.Module = VQModel(**config['model']['params']).to(device)

    print('Num parameters:', sum(p.numel() for p in vae.parameters() if p.requires_grad))

    # optimizer and criterion
    optimizer = torch.optim.Adam(lr=config['model']['base_learning_rate'], params=vae.parameters())
    criterion = nn.MSELoss()

    # lr scheduler
    scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    scheduler_warum_up = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, total_iters=200)

    # login to wandb
    wandb.login(key='e5ef4f3a1142de13823dd7b320a9e133b3f5bdfc')

    best_loss = torch.tensor(99999999.0).to(device)
    best_model = None

    os.makedirs('./weights', exist_ok=True)
    os.makedirs('./plots', exist_ok=True)

    EPOCHS = 1
    total_steps = 0
    for epoch in range(EPOCHS):
        print('Epoch:', epoch)
        # training
        vae.train()
        train_loss = torch.tensor(0).to(device)
        train_steps = 0
        for i, data in tqdm(enumerate(train_data), total=len(train_data), desc='[train]'):
            spec = data['spectrogram'].to(device)

            # forward pass
            reconstructed = vae.forward(spec)

            # loss
            loss = criterion(reconstructed, target=spec)

            # optim
            loss.backward()
            optimizer.step()

            wandb.log({'step': i, 'train_step_loss': loss})

            scheduler_warum_up.step(epoch)
            train_loss += loss
            train_steps += 1

        train_loss = train_loss/train_steps

        vae.test()
        test_loss = torch.tensor(0).to(device)
        test_steps = 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(test_data), total=len(test_data), desc='[test]'):
                spec = data['spectogram'].to(device)

                # forward pass
                reconstructed = vae.forward(spec)

                # loss
                loss = criterion(reconstructed, target=spec)

                wandb.log({'step': i, 'test_step_loss': loss})

                test_loss += loss
                test_steps += 1

        test_loss = test_loss/test_steps

        # validate
        scheduler_plateau.step(test_loss)

        if test_loss < best_loss:
            print('Best Accuracy, saving model')
            best_model = test_loss
            torch.save(vae, os.path.join('./weights', f'best_train_model.pt'))

        # saving spectograms after each epoch for visualization
        fig, axs = plt.subplots(2, 2*5, figsize=(10, 10))
        fig.subtitle("Spectrogram (db)")
        for i in range(5):
            sample = dataset_test[i]

            reconstructed = vae.forward(sample)

            left_og = torch.view_as_complex(sample[[0, 1], :, :].view(513, -1, 2))
            right_og = torch.view_as_complex(sample[[2, 3], :, :].view(513, -1, 2))
            left_pred = torch.view_as_complex(sample[[0, 1], :, :].view(513, -1, 2))
            right_pred = torch.view_as_complex(sample[[2, 3], :, :].view(513, -1, 2))

            axs[0][i*2].set_title('OG - Left')
            axs[0][i*2].imshow(librosa.power_to_db(left_og), origin="lower", aspect="auto")
            axs[0][i*2+1].set_title('OG - Right')
            axs[0][i*2+1].imshow(librosa.power_to_db(right_og), origin="lower", aspect="auto")

            axs[1][i*2].set_title('Pred - Left')
            axs[1][i*2].imshow(librosa.power_to_db(left_pred), origin="lower", aspect="auto")
            axs[1][i*2+1].set_title('Pred - Right')
            axs[1][i*2+1].imshow(librosa.power_to_db(right_pred), origin="lower", aspect="auto")


        path_plot = os.path.join('./plots', f'{epoch}_plot.png')
        fig.savefig(path_plot, dpi=200)
        fig.clear()
        plt.close(fig)

        image = wandb.Image(path_plot, caption="Test_plot")
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'plot': fig
        })

        # save_best_model
        torch.save(vae, os.path.join('./weights', f'epoch_{epoch}.pt'))


if __name__ == '__main__':
    main()