from sklearn.metrics import mean_squared_error
import pickle
from sklearn import metrics
import pandas as pd
import xgboost as xgb



def get_data():
    X_train = pd.read_csv('./data/train_norm.csv')
    X_val = pd.read_csv('./data/val_norm.csv')
    X_test = pd.read_csv('./data/test_norm.csv')

    cols = [col for col in X_train.columns if col != 'rooms']  # remove rooms
    X_train = X_train[cols]
    X_val = X_val[cols]
    X_test = X_test[cols]

    y_train = X_train['price']
    X_train.drop(columns=['price'], inplace=True)
    y_val = X_val['price']
    X_val.drop(columns=['price'], inplace=True)
    y_test = X_test['price']
    X_test.drop(columns=['price'], inplace=True)

    return X_train, y_train, X_val, y_val, X_test, y_test


def model_predict(model, xgboost_model=False):
    X_train, y_train, X_val, y_val, X_test, y_test = get_data()

    if xgboost_model:
        X_train = xgb.DMatrix(X_train, label=y_train)
        X_val = xgb.DMatrix(X_val, label=y_val)
        X_test = xgb.DMatrix(X_test, label=y_test)

    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    target_scaler = pickle.load(open('./utils/target_scaler.pkl', 'rb'))

    y_train_pred = target_scaler.inverse_transform(y_train_pred[:, None])
    y_val_pred = target_scaler.inverse_transform(y_val_pred[:, None])
    y_test_pred = target_scaler.inverse_transform(y_test_pred[:, None])

    y_train_inverse = target_scaler.inverse_transform(y_train.to_numpy()[:, None])
    y_val_inverse = target_scaler.inverse_transform(y_val.to_numpy()[:, None])
    y_test_inverse = target_scaler.inverse_transform(y_test.to_numpy()[:, None])

    return y_train_pred, y_train_inverse, y_val_pred, y_val_inverse, y_test_pred, y_test_inverse


def get_metrics(model, xgboost_model=False):
    y_train_pred, y_train_inverse, y_val_pred, y_val_inverse, y_test_pred, y_test_inverse = model_predict(model,
                                                                                                          xgboost_model)

    rmse_train = mean_squared_error(y_train_inverse, y_train_pred, squared=False)
    rmse_val = mean_squared_error(y_val_inverse, y_val_pred, squared=False)
    rmse_test = mean_squared_error(y_test_inverse, y_test_pred, squared=False)

    r2_train = metrics.r2_score(y_train_inverse, y_train_pred)
    r2_val = metrics.r2_score(y_val_inverse, y_val_pred)
    r2_test = metrics.r2_score(y_test_inverse, y_test_pred)

    res = dict(rmse_train=rmse_train, rmse_val=rmse_val, rmse_test=rmse_test, r2_train=r2_train, r2_val=r2_val,
               r2_test=r2_test)

    return res
