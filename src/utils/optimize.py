import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

def correlation(filename):
    try:
        print('\n🔆 INIT CSV FILE')
        data = pd.read_csv(filename)
        print(f'\n🟢 File "{filename}" loaded successfully\n')
    except FileNotFoundError:
        print('❌ Error: File not found')
        exit(1)
    except pd.errors.EmptyDataError:
        print('❌ Error: Dataset file is empty')
        exit(1)
    except pd.errors.ParserError:
        print('❌ Error: Invalid CSV file format.')
        exit(1)
    except Exception as e:
        print('❌ Error:', e)
        exit(1)

    # Splitting features and target variable
    X = data[['Astronomy', 'History of Magic', 'Transfiguration', 'Charms', 'Flying',
              'Arithmancy', 'Care of Magical Creatures', 'Herbology', 'Potions',
              'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes']]
    y = data['Hogwarts House']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Imputar valores faltantes con la media de cada columna
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train_imputed, y_train)

    # Calculate accuracy before adding the feature
    y_pred_before = model.predict(X_test_imputed)
    accuracy_before = accuracy_score(y_test, y_pred_before)
    print(f'Accuracy before adding the feature: {accuracy_before:.2f}')

    correlation_matrix = X_train.corr()

    product_correlation = correlation_matrix.abs().prod()
    variable_max_corr_product = product_correlation.idxmax()
    print(f'\n🏆 Feature with highest product of absolute correlation coefficients: {variable_max_corr_product}\n')

    # Add the feature with highest correlation
    X_train_max_corr = X_train_imputed.copy()
    # X_train_max_corr = pd.concat([X_train_max_corr, data[variable_max_corr_product]], axis=1)
    X_train_max_corr = pd.DataFrame(X_train_max_corr, columns=X.columns)
    X_train_max_corr = pd.concat([X_train_max_corr, data[variable_max_corr_product]], axis=1)

    # Train a logistic regression model with the added feature
    model_with_max_corr = LogisticRegression()
    model_with_max_corr.fit(X_train_max_corr, y_train)

    # Calculate accuracy after adding the feature
    X_test_max_corr = X_test_imputed.copy()
    X_test_max_corr = pd.concat([X_test_max_corr, data.loc[X_test.index, variable_max_corr_product]], axis=1)
    y_pred_after = model_with_max_corr.predict(X_test_max_corr)
    accuracy_after = accuracy_score(y_test, y_pred_after)
    print(f'Accuracy after adding the feature: {accuracy_after:.2f}')


def main():
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 2:
        print('❗️ Usage: python3 logreg_train.py data*.csv')
        exit(1)

    # Get the file path from command line arguments
    file_path = sys.argv[1]
    print(f'\n🟡 File path:{file_path}\n')

    # Check if the file exists
    if not os.path.isfile(file_path):
        print('❌ Error: File not found')
        exit(1)

    correlation(file_path)


if __name__ == "__main__":
    main()
