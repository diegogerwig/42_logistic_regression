# 42_logistic_regression

## Usage

*Show dataset stats and metrics*
```sh
python3 describe.py data/dataset_train.csv
```

*Draw histogram*
```sh
python3 histogram.py data/dataset_train.csv
```

*Draw scatter plot*
```sh
python3 scatter_plot.py data/dataset_train⚙️.csv
```

*Draw pair plot*
```sh
python3 pair_plot.py data/dataset_train.csv
```

## 🚀 TRAIN PROCRESS
1.  Parser dataset path.
2.  Read dataset from CSV file.
3.  Get numeric features from dataset (making a copy from the original dataset).
4.  Replace NaN data with median value.
5.  Drop non relevant features.
6.  Transform dataframe into a numpy array.
7.  Nomalize data.
8.  Create array with 'y' data (Hogwarts house).
9.  Init 'thetas' with 0. (Number of thetas is equal to number of features + 1 'bias term').
10. Train
11. Save thetas in a CSV file

*   🏁 After the training:
    *  Calculate the accuracy in the prediction
    *  Plot loss history



## 💡 PREDICT PROCRESS
1. 
2. 


## INFO
+ https://www.youtube.com/playlist?list=PLblh5JKOoLUKxzEP5HA2d-Li7IJkHfXSe
+ https://www.youtube.com/watch?v=YYEJ_GUguHw&t=13s
