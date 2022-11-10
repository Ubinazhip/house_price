# house_price
## Task
Predict the prices of flats in Almaty(Kazakhstan).
## Data
Dataset [Data.xlsx](https://github.com/Ubinazhip/house_price/blob/main/Data.xlsx) was derived from the popular website in Kazakhstan krisha.kz. It has features like rooms, sq_m, floor, floors_all, year nad target variable price
## Files
- [notebook.ipynb](https://github.com/Ubinazhip/house_price/blob/main/notebook.ipynb) - data analysis, clearning, train val test split, finetuning ridge and lasso regression and finetuning xgboost model using hyperopt <br>
- [preprocessing.py](https://github.com/Ubinazhip/house_price/blob/main/preprocess.py) - all preprocessing steps, train test split in one file. <br> 
- [train.py](https://github.com/Ubinazhip/house_price/blob/main/train.py) - training xgboost model(which was found to be the best among ridge and lasso) using best parameters, which were found during the finetuning in notebook.ipynb <br>
- [predict.py](https://github.com/Ubinazhip/house_price/blob/main/predict.py) - loads xgboost model and predicts. The flask application is used. <br>
- [test.py](https://github.com/Ubinazhip/house_price/blob/main/test.py) - after you run predict file, you can send the requests using test.py file <br>
- [data](https://github.com/Ubinazhip/house_price/tree/main/data) - folder with train, val and test splits. Derived using [preprocessing.py](https://github.com/Ubinazhip/house_price/blob/main/preprocess.py) inside [notebook.ipynb](https://github.com/Ubinazhip/house_price/blob/main/notebook.ipynb)
## How to run the project
1) clone the project - git clone https://github.com/Ubinazhip/house_price.git  <br>
2) use docker to build an image - **docker build -t house_price .** <br>
3) run the image and you will be in bash - **docker run -it --rm -p 9696:9696 house_price:latest** <br>
4) train the xgboost model - **python3 train.py** <br>
5) model deployement using flask - **python3 predict.py**
6) in another terminal run - **python3 test.py**

## Author
- Aslan Ubingazhibov - aslan.ubingazhibov@alumni.nu.edu.kz
