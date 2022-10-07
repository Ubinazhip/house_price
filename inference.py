import torch
from tqdm import tqdm
import copy
from sklearn.metrics import mean_squared_error as mse
import numpy as np


def inference(model, loader, criterion, target_scaler):
    model.eval()
    tqdm_loader = tqdm(loader)
    running_loss = 0.0
    running_rmse = 0.0
    for batch_idx, (X, y) in enumerate(tqdm_loader):
        X, y = X.cuda(), y.cuda()
        with torch.no_grad():
            pred = model(X)
            loss = criterion(pred, y[:, None].float())
            y_inverse = target_scaler.inverse_transform(y.cpu()[:, None])
            y_pred_inverse = target_scaler.inverse_transform(pred.detach().cpu().numpy())
            running_loss += loss.item() * X.size(0)
            running_rmse += mse(y_inverse, y_pred_inverse) * X.size(0)
            tqdm_loader.set_description(f'Test...')

    epoch_loss = running_loss / len(loader.dataset)
    epoch_rmse = np.sqrt(running_rmse / len(loader.dataset))
    print(f'test: Loss: {epoch_loss:.2f}, rmse = {epoch_rmse:.2f}')
    print('--------------------------------------------------------')
    print('\n')

    return epoch_loss, epoch_rmse