import pickle
import xgboost as xgb
from utils import utils
from preprocess import Preprocessing

data = Preprocessing(file_name="Data.xlsx", test_size=0.1, random_state=20, save_artefacts=True)

X_train, X_val, X_test = data.preprocess_data()

X_train, y_train, X_val, y_val, X_test, y_test = utils.get_data()
train = xgb.DMatrix(X_train, label=y_train)
valid = xgb.DMatrix(X_val, label=y_val)
test = xgb.DMatrix(X_test, label=y_test)

#the best results from finetuning xgboost model in notebook.ipynb

best_result = {'learning_rate': 0.4994846948849469,
               'max_depth': 3,
               'reg_alpha': 0.01512012651698214,
               'reg_lambda': 0.0031891178062977013}

print('training xgboost model .... ')
booster = xgb.train(
    params=best_result,
    dtrain=train,
    evals=[(valid, 'validation')],
    num_boost_round=40,
    early_stopping_rounds=10
)

pickle.dump(booster, open('./utils/xgboost_model.pkl', 'wb'))

print(f'The model is saved as ./utils/xgboost_model.pkl')
print('The results for Xgboost model: ')
xboost_results = utils.get_metrics(booster, xgboost_model=True)
print(xboost_results)