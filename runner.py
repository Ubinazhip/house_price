from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, CosineAnnealingLR
import argparse
import warnings
warnings.filterwarnings("ignore")
import os
from model import get_model
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, CosineAnnealingLR
from data_loader import data_loader
import numpy as np
import pickle
from train import train_model
from inference import inference
import mlflow


def init_seed(SEED=42):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16, help="batch_size")
    parser.add_argument("--fold", type=int, default=0, help="fold")
    parser.add_argument("--epochs", type=int, default=50, help="number of training epochs")
    parser.add_argument('--model', type=str, default='net1', help='model name')
    parser.add_argument('--loss_type', type=str, default='mse', help='mse or mae')
    parser.add_argument('--experiment_name', type=str, default='third_exp', help='experiment name')
    parser.add_argument('--scheduler_patience', type=int, default=5)
    parser.add_argument("--scheduler_factor", type=float, default=0.1, help="scheduler reduce factor")
    parser.add_argument("--weight_decay", type=float, default=5e-5, help="weight decay in adam optimizer")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")

    parser.add_argument("--drop_rate", type=float, default=0.1, help="dropout")
    parser.add_argument('--amsgrad', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser()
    print(f'model = {args.model}, fold={args.fold}, loss={args.loss_type}')
    init_seed(SEED=args.seed)
    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(args.experiment_name)

    target_scaler = pickle.load(open(f'./utils/target_scaler{args.fold}.pkl', 'rb'))
    model = get_model(model_name=args.model, drop_rate=args.drop_rate)
    model = model.cuda()
    dataloaders = data_loader(batch_size=args.batch_size, fold=args.fold)

    if args.loss_type == 'mse':
        criterion = torch.nn.MSELoss()
    elif args.loss_type == 'mae':
        criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=args.amsgrad, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=args.scheduler_patience,
                                  factor=args.scheduler_factor, min_lr=1e-8, verbose=True)

    hparams = dict(test_rmse=0, test_loss=0, val_best_rmse=0, train_best_rmse=0, fold=args.fold,
                   model=args.model, drop_rate=args.drop_rate, batch_size=args.batch_size,
                   num_epochs=args.epochs, loss_type=args.loss_type, lr=args.lr, weight_decay=args.weight_decay,
                   scheduler_patience=args.scheduler_patience, factor=args.scheduler_factor, seed=args.seed)
    with mlflow.start_run():
        model, best_rmse, best_rmse_train = train_model(model, dataloaders, criterion, optimizer, scheduler,
                                                        target_scaler, hparams, mlflow)
        test_loss, test_rmse = inference(model, dataloaders['test'], criterion, target_scaler)
        hparams['val_best_rmse'] = best_rmse
        hparams['train_best_rmse'] = best_rmse_train
        hparams['test_loss'] = test_loss
        hparams['test_rmse'] = test_rmse
        mlflow.log_params(hparams)

    if test_rmse < 17500000:
        model_name = f'{args.model}_fold{args.fold}_train{best_rmse_train:.1f}_val{best_rmse:.1f}_test{test_rmse:.1f}.pth'
        torch.save(model, os.path.join(f'./best_models/{model_name}'))