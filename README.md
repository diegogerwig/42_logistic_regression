# ⚡ 42_logistic_regression

---
--- 

## ⚙️ Usage

#### Show dataset stats and metrics
```sh
python3 describe.py data/dataset_train.csv
```

---

#### Draw histogram
```sh
python3 histogram.py data/dataset_train.csv
```

---

#### Draw scatter plot
```sh
python3 scatter_plot.py data/dataset_train.csv
```

---

#### Draw pair plot
```sh
python3 pair_plot.py data/dataset_train.csv
```

---

#### Train model
```sh
python3 logreg_train.py data/dataset_train.csv
```
> returns 'params.csv' file

---

#### Predict houses
```sh
python3 logreg_predic.py data/dataset_test.csv
```
> needs 'params.csv' file

> returns 'houses.csv' file

---

#### Evaluate prediction
```sh
python3 evaluate.py
```
> needs 'houses.csv' file

> needs 'dataset_truth.csv' file

---

#### Optimize features drop selection
```sh
python3 optimize.py data/dataset_train.csv
```

---
---

## 💥 LINEAR vs LOGISTIC
[![./doc/explanation%20linear%20vs%20logistic.md](./doc/logistic_reg_VS_linear_reg.png)](./doc/explanation%20linear%20vs%20logistic.md)

---
---

## 🚀 TRAIN PROCESS
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

---
---

## 💡 PREDICT PROCESS
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

---
---

## 📊 PLOT EXAMPLES
+ HISTOGRAM
![HISTROGRAM](plots_examples/histogram_with_kde.png)

+ SCATTER
![HISTROGRAM](plots_examples/scatter_plot.png)

+ PAIR PLOT
![HISTROGRAM](plots_examples/pair_plot.png)

+ CORRELATION
![HISTROGRAM](plots_examples/correlation_matrix.png)

+ LOSS HISTORY
![HISTROGRAM](plots_examples/plot_loss_history.png)

+ FEATURE IMPORTANCE
![HISTROGRAM](plots_examples/plot_feature_importance.png)

---
---

## 📖 INFO
+ https://www.youtube.com/playlist?list=PLblh5JKOoLUKxzEP5HA2d-Li7IJkHfXSe
+ https://www.youtube.com/watch?v=YYEJ_GUguHw&t=13s
+ https://www.youtube.com/watch?v=C5268D9t9Ak
+ https://en.wikipedia.org/wiki/Logistic_regression
+ https://datamapu.com/posts/classical_ml/logistic_regression/
+ https://www.youngwonks.com/blog/What-is-sklearn-Logistic-Regression
