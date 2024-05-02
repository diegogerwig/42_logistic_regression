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
1.  Parse dataset path.
2.  Read dataset from CSV file (📝 dataset_train.csv).
3.  Get numeric features from dataset (by making a copy of the original dataset).
4.  Replace NaN values with the median value.
5.  Drop non-relevant features.
6.  Transform the dataframe into a numpy array.
7.  Nomalize the data.
8.  Create an array with 'y' data (Hogwarts house).
9.  Initialize 'thetas' with 0 (the number of thetas is equal to number of features + 1 for the 'bias term').
10. Train
11. Save the 'thetas' in a CSV file (📝 params.csv).

*   🏁 After the train:
    *  Calculate the accuracy of the prediction.
    *  Plot the loss history.


## 💡 PREDICT PROCRESS
1.  Parse dataset path.
2.  Load 'thetas' from CSV file (📝 params.csv).
3.  Check format info in CSV file (📝 dataset_test.csv).
4.  Read dataset from CSV file (📝 dataset_test.csv).
5.  Get numeric features from dataset (by making a copy of the original dataset).
6.  Replace NaN values with the median value.
7.  Drop non-relevant features.
8.  Transform the dataframe into a numpy array.
9.  Nomalize the data.
10. Predict
11. Save the 'predicted houses' in a CSV file (📝 houses.csv).

*   🏁 After the predict:
    *  Evaluate the predicted data with real data.
    *  Plot relevance features.


## INFO
+ https://www.youtube.com/playlist?list=PLblh5JKOoLUKxzEP5HA2d-Li7IJkHfXSe
+ https://www.youtube.com/watch?v=YYEJ_GUguHw&t=13s
