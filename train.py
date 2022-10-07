import torch
from tqdm import tqdm
import copy
from sklearn.metrics import mean_squared_error as mse
import numpy as np


def get_state_dict(model):
    if type(model) == torch.nn.DataParallel:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    return state_dict


def load_model(model, best_model_state_dict):
    if type(model) == torch.nn.DataParallel:
        model.module.load_state_dict(best_model_state_dict)
    else:
        model.load_state_dict(best_model_state_dict)
    print('best model weights are loaded ...')
    print()
    return model


def train_model(model, dataloaders, criterion, optimizer, scheduler, target_scaler, hparams, mlflow):

    best_rmse = float('inf')
    best_rmse_train = 0
    best_model_wts = copy.deepcopy(get_state_dict(model))
    for epoch in range(hparams['num_epochs']):
        running_loss = 0.0
        running_rmse = 0.0
        running_loss_train = 0.0
        running_rmse_train = 0.0

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            tqdm_loader = tqdm(dataloaders[phase])
            for batch_idx, (X, y) in enumerate(tqdm_loader):
                X, y = X.cuda(), y.cuda()
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    pred = model(X)
                    loss = criterion(pred, y[:,None].float())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                y_inverse = target_scaler.inverse_transform(y.cpu()[:,None])
                y_pred_inverse = target_scaler.inverse_transform(pred.detach().cpu().numpy())
                if phase == 'val':
                    running_loss += loss.item() * X.size(0)
                    running_rmse += mse(y_inverse, y_pred_inverse) * X.size(0)
                    tqdm_loader.set_description(f'Validation...')
                else:
                    running_loss_train += loss.item() * X.size(0)
                    running_rmse_train += mse(y_inverse, y_pred_inverse) * X.size(0)
                    tqdm_loader.set_description(f'Training...')
        epoch_loss = running_loss / len(dataloaders['val'].dataset)
        epoch_rmse = np.sqrt(running_rmse / len(dataloaders['val'].dataset))
        epoch_loss_train = running_loss_train / len(dataloaders['train'].dataset)
        epoch_rmse_train = np.sqrt(running_rmse_train / len(dataloaders['train'].dataset))
        metrics = dict(val_loss=epoch_loss, val_rmse=epoch_rmse,
                       train_loss=epoch_loss_train, rmse_train=epoch_rmse_train)
        mlflow.log_metrics(metrics, step=epoch)
        print(f'epoch = {epoch}, train: Loss: {epoch_loss_train:.2f}, rmse = {epoch_rmse_train:.2f}')
        print(f'epoch = {epoch}, valid: Loss: {epoch_loss:.2f}, rmse = {epoch_rmse:.2f}')
        if phase == "val":
            if best_rmse > epoch_rmse:
                best_rmse = epoch_rmse
                best_rmse_train = epoch_rmse_train
                best_model_wts = copy.deepcopy(get_state_dict(model))
                print('Best so far')
            scheduler.step(epoch_loss)

    model.load_state_dict(best_model_wts)
    print(f'best model in terms of val rmse: val rmse = {best_rmse:.2f}, train: {best_rmse_train:.3f}')
    return model, best_rmse, best_rmse_train