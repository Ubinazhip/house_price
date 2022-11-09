import flask
import pandas as pd
import pickle


almaty_areas = ['Ауэзовский', 'Бостандыкский', 'Алмалинский', 'Алатауский', 'Медеуский', 'Наурызбайский', 'Турксибский',
                'Жетысуский']

target_scaler = pickle.load(open('./utils/target_scaler.pkl','rb'))
feature_scaler = pickle.load(open('./utils/feature_scaler.pkl','rb'))
one_hot_enc = pickle.load(open('./utils/one_hot_enc.pkl','rb'))

year_median_floor = pickle.load(open('./utils/year_median_floor.pkl','rb'))
area_median_year = pickle.load(open('./utils/area_median_year.pkl','rb'))


def streets_to_area(curr_area):
    if curr_area.item() not in almaty_areas:
        return 'district9'
    return curr_area.item()


def year_to_int(df):
    try:
        curr_year = int(df['year'])
    except:
        curr_year = area_median_year[df['area'].item()]
    return curr_year


def fill_empty_floor(df):
    if df['floor'].item() is None and df['floors_all'].item() is not None:
        df['floor'] = max(df['floors_all'].item() // 2, 1)
    elif df['floor'].item() is not None and df['floors_all'].item() is None:
        df['floors_all'] = df['floor'] * 2
    elif df['floor'].item() is None and df['floors_all'].item() is None:
        if df['year'].item() <= 1990:
            df['floors_all'] = year_median_floor['[ - 1990]']
        elif df['year'].item() >= 2010:
            df['floors_all'] = year_median_floor['[2010 - )']
        else:
            df['floors_all'] = year_median_floor['(1990 - 2010]']
        df['floor'] = df['floors_all'] // 2
    return df


def one_hot(df):
    res = one_hot_enc.transform(df['area'].to_numpy()[:, None]).toarray()
    cols_name = [f'area{i}' for i in range(1, 9)]
    df_train_one_hot = pd.DataFrame(res, columns=cols_name, dtype=int)
    df = pd.concat([df, df_train_one_hot], axis=1)
    df.drop(columns=['area'], inplace=True)
    return df


def scale_data(df):
    numeric_features = ['rooms', 'sq_m', 'floor', 'floors_all', 'year']
    df[numeric_features] = feature_scaler.transform(df[numeric_features])


def preprocess(curr_dict):
    df = pd.DataFrame(curr_dict, index=[0])
    df['area'] = streets_to_area(df['area'])
    df['year'] = year_to_int(df)
    df = fill_empty_floor(df)
    #print(df['floors_all'])
    scale_data(df)
    df = one_hot(df)
    return df


if __name__ == '__main__':
    curr_dict = {'rooms': 2,
                 'sq_m': 76.0,
                 'floor': 10,
                 'floors_all': 10,
                 'area': 'Алмалинский',
                 'year': '2020'}
    res = preprocess(curr_dict)
    model1 = pickle.load(open('./utils/ridge_model.pkl','rb'))
    model2 = pickle.load(open('./utils/xgboost_model.pkl','rb'))
    res = res.drop(columns=["rooms"])
    y_pred_ridge = model1.predict(res)
    y_pred_xgboost = model2.predict(res)

    price_ridge = target_scaler.inverse_transform(y_pred_ridge[:,None])
    price_xgboost = target_scaler.inverse_transform(y_pred_xgboost[:,None])
    print(f'the house characteristics {curr_dict}')
    print(f'The price for the given house is:')
    print(f'Ridge prediction: =  {price_ridge[0][0]:.1f}')
    print(f'Xgboost prediction: =  {price_xgboost[0][0]:.1f}')
