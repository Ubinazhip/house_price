# task_house_price

1) Preprocessing описан в ipynb файле Preprocess_data и реализован в preprocess.py <br>
2) Поделил датасет на трейн(0.7), val(0.2) и test(0.1). Трейн и вал - 5 фолдов <br>
3) Использовал нейронку и xgboost(c hyperopt), но у нейронки рез-ы получше <br>
4) experiment tracking с помощью mlflow <br>
5) Построил 6 разных не больших Fully Connected Layers моделей, веса были инициализованы по He initializtion. <br>
6) Сравниваю модели с помощью RMSE <br>
7) Вторая задача в ipynb файле task2
8) Лучшие модели [здесь](https://drive.google.com/drive/folders/13x2UTMFW1b_NonXI6f55mTAWRKXkb-6t?usp=sharing)
9) лучшая модель показывает rmse train 18.5million, val=18.2 million, test=17.2 million. Можно было бы больше поэкспериментировать с припросессенгами, моделями, нейронки vs деревья, но из-за нехватки времени и железа пока так. <br>
10) Рез-ы эспериментов можно посмотреть [здесь](https://drive.google.com/drive/folders/13x2UTMFW1b_NonXI6f55mTAWRKXkb-6t?usp=sharing), либо через mlflow.db
11) В run5.sh можете посмотреть как ранить тренировку модели
