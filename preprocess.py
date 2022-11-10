import pandas as pd
import warnings
warnings.simplefilter("ignore")
import math
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
almaty_areas = ['Ауэзовский', 'Бостандыкский', 'Алмалинский', 'Алатауский', 'Медеуский', 'Наурызбайский', 'Турксибский',
                'Жетысуский']
import pickle
from sklearn.model_selection import train_test_split


class Preprocessing():
    def __init__(self, file_name="Data.xlsx", test_size=0.1, val_size=0.2, random_state=42, save_artefacts=False):
        self.df = pd.read_excel(file_name)
        self.df = self.df.drop(columns=["urls"])
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        y = self.df['price']
        self.df = self.df.drop(columns=['price'])

        X_train_val, self.X_test, y_train_val, self.y_test = train_test_split(self.df, y, test_size=self.test_size,
                                                                              random_state=self.random_state)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_train_val, y_train_val, test_size=self.val_size,
                                                                              random_state=self.random_state)
        self.X_train = pd.concat([self.X_train, self.y_train], axis=1)
        self.X_val = pd.concat([self.X_val, self.y_val], axis=1)
        self.X_test = pd.concat([self.X_test, self.y_test], axis=1)
        self.save_artefacts = save_artefacts

    def streets_to_area(self, dfX):
        '''
        convert street names in area into new area "Район9"
        :param dfX: dataframe
        '''
        dfX['area'] = dfX['area'].apply(lambda x: x if x in almaty_areas else 'district9')
        return dfX

    def year_to_int(self, dfX, train=False):
        '''
        cast year column to int, some entries are non-convertible,
        so drop during the training, and fill with median during
        the test and val
        :param dfX: data
        :param train: whether dfX, dfy is train set
        '''
        empty_strs = [' ', '  ', '   ', '    ']
        empty_str_ind = dfX.loc[dfX.year.isin(empty_strs)].index.values
        if train:
            dfX.drop(empty_str_ind, inplace=True)
            dfX['year'] = dfX['year'].astype(int)
            self.area_median_year = dfX.groupby(['area'])['year'].median().to_dict()
        else:
            for i in range(0, dfX.shape[0]):
                dfX.year.iloc[i] = self.area_median_year[dfX.area.iloc[i]]
        if self.save_artefacts:
            with open('./utils/area_median_year.pkl', 'wb') as f:
                pickle.dump(self.area_median_year, f)

        return dfX

    def missing_entries(self, dfX, train=False):
        '''

        fill missing entries in floors and floors_all

        '''
        if train:
            year_div = []
            for i in range(0, dfX.shape[0]):
                if dfX.iloc[i]['year'] <= 1990:
                    year_div.append('[ - 1990]')
                elif 1990 < dfX.iloc[i]['year'] <= 2010:
                    year_div.append('(1990 - 2010]')
                else:
                    year_div.append('[2010 - )')
            dfX['year_div'] = year_div
            self.year_median_floor = dfX.groupby('year_div')['floors_all'].median().to_dict()
            dfX = dfX.drop(columns=["year_div"])

        for i in range(0, dfX.shape[0]):
            if math.isnan(dfX.iloc[i]['floors_all']) and math.isnan(dfX.iloc[i]['floor']):
                dfX['floors_all'].iloc[i] = self.find_floor_all(dfX.iloc[i])
                dfX['floor'].iloc[i] = dfX['floors_all'].iloc[i] // 2
            elif math.isnan(dfX.iloc[i]['floors_all']) and not math.isnan(dfX.iloc[i]['floor']):
                dfX['floors_all'].iloc[i] = 2 * dfX['floor'].iloc[i]
            elif math.isnan(dfX.iloc[i]['floor']) and not math.isnan(dfX.iloc[i]['floors_all']):
                if dfX['floors_all'].iloc[i] == 1:
                    dfX['floors'].iloc[i] = 1
                else:
                    dfX['floor'].iloc[i] = dfX['floors_all'].iloc[i] // 2
        if self.save_artefacts:
            with open('./utils/year_median_floor.pkl', 'wb') as f:
                pickle.dump(self.year_median_floor, f)
        return dfX

    def find_floor_all(self, row):
        if row['year'] <= 1990:
            floors_all = self.year_median_floor['[ - 1990]']
        elif 1990 < row['year'] <= 2010:
            floors_all = self.year_median_floor['(1990 - 2010]']
        else:
            floors_all = self.year_median_floor['[2010 - )']
        return floors_all

    def reset_indices(self):
        self.X_train.reset_index(drop=True, inplace=True)
        self.X_val.reset_index(drop=True, inplace=True)
        self.X_test.reset_index(drop=True, inplace=True)

    def one_hot_encoder(self):
        '''

        one hot encoder of column area

        '''
        one_hot = OneHotEncoder(drop='first')
        one_hot.fit(self.X_train['area'][:, None])
        arr_train = one_hot.transform(self.X_train['area'][:, None]).toarray()
        arr_val = one_hot.transform(self.X_val['area'][:, None]).toarray()
        arr_test = one_hot.transform(self.X_test['area'][:, None]).toarray()
        cols_name = ['area1', 'area2', 'area3', 'area4', 'area5', 'area6', 'area7', 'area8']
        df_train_one_hot = pd.DataFrame(arr_train, columns=cols_name, dtype=int)
        df_val_one_hot = pd.DataFrame(arr_val, columns=cols_name, dtype=int)
        df_test_one_hot = pd.DataFrame(arr_test, columns=cols_name, dtype=int)
        self.X_train = pd.concat([self.X_train, df_train_one_hot], axis=1)
        self.X_val = pd.concat([self.X_val, df_val_one_hot], axis=1)
        self.X_test = pd.concat([self.X_test, df_test_one_hot], axis=1)
        self.X_train.drop(columns=['area'], inplace=True)
        self.X_val.drop(columns=['area'], inplace=True)
        self.X_test.drop(columns=['area'], inplace=True)
        if self.save_artefacts:
            pickle.dump(one_hot, open(f'./utils/one_hot_enc.pkl', 'wb'))

    def scale_data(self):
        '''

        normalize numeric features and target variable

        '''
        numeric_features = ['rooms', 'sq_m', 'floor', 'floors_all', 'year']
        scaler = StandardScaler()
        self.X_train[numeric_features] = scaler.fit_transform(self.X_train[numeric_features])
        self.X_val[numeric_features] = scaler.transform(self.X_val[numeric_features])
        self.X_test[numeric_features] = scaler.transform(self.X_test[numeric_features])
        scaler_target = StandardScaler()
        self.X_train['price'] = scaler_target.fit_transform(self.X_train['price'][:,None])
        self.X_val['price'] = scaler_target.transform(self.X_val['price'][:,None])
        self.X_test['price'] = scaler_target.transform(self.X_test['price'][:,None])

        if self.save_artefacts:
            pickle.dump(scaler, open(f'./utils/feature_scaler.pkl', 'wb'))
            pickle.dump(scaler_target, open(f'./utils/target_scaler.pkl', 'wb'))

    def preprocess_data(self):
        '''

        overall preprocessing steps:
        1-step: convert street names to district9
        2-step: convert year of type string to int
        3-step: fill missing entries

        '''
        self.X_train = self.streets_to_area(self.X_train)
        self.X_val = self.streets_to_area(self.X_val)
        self.X_test = self.streets_to_area(self.X_test)
        self.X_train = self.year_to_int(self.X_train, train=True)
        self.X_val = self.year_to_int(self.X_val, train=False)
        self.X_test = self.year_to_int(self.X_test, train=False)
        self.X_train = self.missing_entries(self.X_train, train=True)
        self.X_val = self.missing_entries(self.X_val, train=False)
        self.X_test = self.missing_entries(self.X_test, train=False)
        self.reset_indices()
        self.one_hot_encoder()
        if self.save_artefacts:
            self.X_train.to_csv(f'./data/train.csv', index=False)#pd.concat([self.X_train, self.y_train], axis=1)
            self.X_val.to_csv(f'./data/val.csv', index=False)
            self.X_test.to_csv(f'./data/test.csv', index=False)
        self.scale_data()
        if self.save_artefacts:
            self.X_train.to_csv(f'./data/train_norm.csv', index=False)#pd.concat([self.X_train, self.y_train], axis=1)
            self.X_val.to_csv(f'./data/val_norm.csv', index=False)
            self.X_test.to_csv(f'./data/test_norm.csv', index=False)

        return self.X_train, self.X_val, self.X_test
