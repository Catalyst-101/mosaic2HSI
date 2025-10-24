# import os
# import yaml
# import argparse
# from torch.utils.data import DataLoader
# import torch
# from torch import optim
# from tqdm import tqdm
# from data.dataset import HSIDataset
# from models.unet import UNet
# from metrics import l1_loss, spectral_angle_mapper
# from torch.utils.tensorboard import SummaryWriter


# def save_checkpoint(state, path):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     torch.save(state, path)


# def train(cfg_path):
#     cfg = yaml.safe_load(open(cfg_path))
#     cfg['train']['lr'] = float(cfg['train']['lr'])
#     cfg['train']['weight_decay'] = float(cfg['train']['weight_decay'])
#     train_ds = HSIDataset(cfg['data']['train_dir'], patch_size=cfg['data']['patch_size'], is_train=True)
#     val_ds = HSIDataset(cfg['data']['val_dir'], patch_size=cfg['data']['patch_size'], is_train=False)
#     train_loader = DataLoader(train_ds, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=cfg['data']['num_workers'])
#     val_loader = DataLoader(val_ds, batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=cfg['data']['num_workers'])

#     device = torch.device(cfg['train']['device'] if torch.cuda.is_available() else 'cpu')
#     model = UNet(in_channels=cfg['model']['in_channels'], out_channels=cfg['model']['out_channels'], base_filters=cfg['model']['base_filters']).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])

#     writer = SummaryWriter(log_dir=os.path.join(cfg['train']['save_dir'], 'logs'))

#     best_val = 1e9
#     for epoch in range(cfg['train']['epochs']):
#         model.train()
#         running_loss = 0.0
#         for mosaic, hsi in tqdm(train_loader, desc=f'Train E{epoch}'):
#             mosaic = mosaic.to(device)
#             hsi = hsi.to(device)
#             pred = model(mosaic)
#             loss = l1_loss(pred, hsi)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#         avg_train = running_loss / len(train_loader)
#         writer.add_scalar('train/l1', avg_train, epoch)

#         # validation
#         model.eval()
#         val_loss = 0.0
#         val_sam = 0.0
#         with torch.no_grad():
#             for mosaic, hsi in val_loader:
#                 mosaic = mosaic.to(device)
#                 hsi = hsi.to(device)
#                 pred = model(mosaic)
#                 val_loss += l1_loss(pred, hsi).item()
#                 val_sam += spectral_angle_mapper(pred, hsi).item()
#         avg_val = val_loss / len(val_loader)
#         avg_sam = val_sam / len(val_loader)
#         writer.add_scalar('val/l1', avg_val, epoch)
#         writer.add_scalar('val/sam', avg_sam, epoch)

#         print(f'Epoch {epoch}: train_l1={avg_train:.6f} val_l1={avg_val:.6f} val_sam={avg_sam:.6f}')

#         # checkpoint
#         ckpt_path = os.path.join(cfg['train']['save_dir'], f'epoch_{epoch:03d}.pth')
#         save_checkpoint({'epoch': epoch, 'model': model.state_dict(), 'opt': optimizer.state_dict()}, ckpt_path)
#         if avg_val < best_val:
#             best_val = avg_val
#             save_checkpoint({'epoch': epoch, 'model': model.state_dict(), 'opt': optimizer.state_dict()}, os.path.join(cfg['train']['save_dir'], 'best.pth'))

#     writer.close()


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', default='configs/default.yaml')
#     args = parser.parse_args()
#     train(args.config)

import os
import yaml
import argparse
from torch.utils.data import DataLoader
import torch
from torch import optim
from tqdm import tqdm
from data.dataset import HSIDataset
from metrics import l1_loss, spectral_angle_mapper
from torch.utils.tensorboard import SummaryWriter

# ✅ Import both models
from models.unet import UNet
from models.cnn import CNNReconstruct


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def train(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    cfg['train']['lr'] = float(cfg['train']['lr'])
    cfg['train']['weight_decay'] = float(cfg['train']['weight_decay'])

    # Load datasets
    train_ds = HSIDataset(cfg['data']['train_dir'],
                          patch_size=cfg['data']['patch_size'],
                          is_train=True)
    val_ds = HSIDataset(cfg['data']['val_dir'],
                        patch_size=cfg['data']['patch_size'],
                        is_train=False)

    train_loader = DataLoader(train_ds,
                              batch_size=cfg['data']['batch_size'],
                              shuffle=True,
                              num_workers=cfg['data']['num_workers'])
    val_loader = DataLoader(val_ds,
                            batch_size=cfg['data']['batch_size'],
                            shuffle=False,
                            num_workers=cfg['data']['num_workers'])

    device = torch.device(cfg['train']['device'] if torch.cuda.is_available() else 'cpu')

    # ✅ Choose model type from config
    model_type = cfg['model'].get('type', 'unet').lower()
    if model_type == 'cnn':
        model = CNNReconstruct(
            in_channels=cfg['model']['in_channels'],
            out_channels=cfg['model']['out_channels']
        ).to(device)
        print("✅ Using CNNReconstruct model")
    else:
        model = UNet(
            in_channels=cfg['model']['in_channels'],
            out_channels=cfg['model']['out_channels'],
            base_filters=cfg['model']['base_filters']
        ).to(device)
        print("✅ Using UNet model")

    optimizer = optim.Adam(model.parameters(),
                           lr=cfg['train']['lr'],
                           weight_decay=cfg['train']['weight_decay'])

    writer = SummaryWriter(log_dir=os.path.join(cfg['train']['save_dir'], 'logs'))

    best_val = 1e9
    for epoch in range(cfg['train']['epochs']):
        model.train()
        running_loss = 0.0

        for mosaic, hsi in tqdm(train_loader, desc=f'Train E{epoch}'):
            mosaic, hsi = mosaic.to(device), hsi.to(device)
            pred = model(mosaic)
            loss = l1_loss(pred, hsi)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train = running_loss / len(train_loader)
        writer.add_scalar('train/l1', avg_train, epoch)

        # Validation
        model.eval()
        val_loss, val_sam = 0.0, 0.0
        with torch.no_grad():
            for mosaic, hsi in val_loader:
                mosaic, hsi = mosaic.to(device), hsi.to(device)
                pred = model(mosaic)
                val_loss += l1_loss(pred, hsi).item()
                val_sam += spectral_angle_mapper(pred, hsi).item()

        avg_val = val_loss / len(val_loader)
        avg_sam = val_sam / len(val_loader)

        writer.add_scalar('val/l1', avg_val, epoch)
        writer.add_scalar('val/sam', avg_sam, epoch)

        print(f"Epoch {epoch}: train_l1={avg_train:.6f} val_l1={avg_val:.6f} val_sam={avg_sam:.6f}")

        # Checkpoint saving
        ckpt_path = os.path.join(cfg['train']['save_dir'], f'epoch_{epoch:03d}.pth')
        save_checkpoint({'epoch': epoch,
                         'model': model.state_dict(),
                         'opt': optimizer.state_dict()}, ckpt_path)

        if avg_val < best_val:
            best_val = avg_val
            save_checkpoint({'epoch': epoch,
                             'model': model.state_dict(),
                             'opt': optimizer.state_dict()},
                            os.path.join(cfg['train']['save_dir'], 'best.pth'))

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()
    train(args.config)
