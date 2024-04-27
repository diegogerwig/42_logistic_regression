import os
import sys
import csv

import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(os.path.dirname(current_dir), 'utils')
sys.path.append(module_dir)
from logreg_train import columns_to_drop, houses, PARAMS_FILE_PATH
from stats import percentile
from normalize import normalize_xset
from gradient import sigmoid

HOUSES_FILE_PATH = 'data/houses.csv'


def predict(filename, thetas):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(filename)
        print('\nINIT CSV FILE')
        df.info()
        print(df)
        input('\nPress Enter to continue...\n')

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


    col_names = ["Index", "Hogwarts House", "First Name", "Last Name",
                    "Birthday", "Best Hand", "Arithmancy", "Astronomy",
                    "Herbology", "Defense Against the Dark Arts",
                    "Divination", "Muggle Studies", "Ancient Runes",
                    "History of Magic", "Transfiguration", "Potions",
                    "Care of Magical Creatures", "Charms", "Flying"]
    col_types = [int, None, object, object, object, object, float, float,
                float, float, float, float, float, float, float, float, float,
                float, float]
    col_check = zip(col_names, col_types)

    # check that the expected columns are here and check their type
    if not set(col_names).issubset(df.columns):
        print(f"Missing columns in '{filename}' file", file=sys.stderr)
        exit(1)
    for name, type_ in col_check:
        if not df[name].dtype == type_:
            print(f"Wrong column type in '{filename} file", file=sys.stderr)
            exit(1)

    # Get numeric features
    df_num = df.select_dtypes(include=['int', 'float']).copy()
    print('\nGET NUMERIC FEATURES')
    df_num.info()
    print(df_num)
    input('\nPress Enter to continue...\n')

    # Replace NaN data with mean value
    for column in df_num.columns:
        median = percentile(df_num[column], 0.50)
        df_num[column] = df_num[column].fillna(median)
    print('\nREPLACE NaN DATA WITH MEDIAN VALUE')
    df_num.info()
    print(df_num)
    input('\nPress Enter to continue...\n')
    
    # df_num.insert(1, 'Hogwarts House', df['Hogwarts House'])
    # print('\nINSERT HOGWARTS HOUSE COLUMN')
    # df_num.info()
    # print(df_num)
    # input('\nPress Enter to continue...\n')

    # Drop category features
    df_num.drop(columns_to_drop, inplace=True, axis=1)
    print('\nREMOVE SOME CATEGORY FEATURES')
    print(f'COLUMNS DROPPED: {columns_to_drop}')
    df_num.info()
    print(df_num)
    input('\nPress Enter to continue...\n')

    # nb_features = len(df_num.columns) - 2
    df_num_excl_first_two = df_num.iloc[:, 2:]
    print('\nREMOVE FIRST TWO COLUMNS')
    df_num_excl_first_two.info()
    print(df_num_excl_first_two)
    input('\nPress Enter to continue...\n')

    nb_features = len(df_num_excl_first_two.columns)
    column_names = df_num_excl_first_two.columns.tolist()
    print(f'\nNumber of features: {nb_features}')
    print(column_names)
    input('\nPress Enter to continue...\n')


    # Normalize the data
    x = np.array(df_num_excl_first_two)
    X_norm = normalize_xset(x)

    # Add bias term to normalized features
    X_norm = np.hstack((np.ones((X_norm.shape[0], 1)), X_norm))

    # Perform predictions
    predictions = []
    for theta in thetas:
        probability = sigmoid(np.dot(X_norm, theta))
        predictions.append(probability)
    print('\nPredictions:')
    np.set_printoptions(suppress=True, threshold=np.inf)
    stacked_predictions = np.column_stack(predictions)
    for pred in predictions:
        print(len(pred))
    print(stacked_predictions)
    input('\nPress Enter to continue...\n')

    # Determine predicted house for each instance
    # Calcular el índice del valor máximo en cada conjunto de cuatro valores
    predicted_house_indices = np.argmax(stacked_predictions, axis=1)

    # Crear una lista para almacenar las casas predichas
    predicted_houses = []

    # Iterar sobre los índices calculados y asignar la casa correspondiente
    for idx in predicted_house_indices:
        predicted_houses.append(houses[idx])

    # Imprimir las casas predichas
    print('\nCasas predichas:')
    print(predicted_houses)

    return predicted_houses


def save_predictions_to_csv(final_predictions):
    # Save the predictions to a CSV file
    try:
        with open(HOUSES_FILE_PATH, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Index", "Hogwarts House"])
            for index, prediction in enumerate(final_predictions):
                writer.writerow([index, prediction])
        print('\n⚪️ Prediction file saved as: {}\n'.format(HOUSES_FILE_PATH))
    except FileNotFoundError:
        print('❌ Error: File not found {}'.format(HOUSES_FILE_PATH),
              file=sys.stderr)
        exit(1)
    except Exception as e:
        print('❌ Error:', e, file=sys.stderr)
        exit(1)


def main():
    '''
    Main function of the program.
    '''
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

    # Load trained parameters
    try:
        with open(PARAMS_FILE_PATH, 'r') as file:
            reader = csv.reader(file)
            thetas = [np.array([float(val) for val in row[0].split(',')]) for row in reader]
        print('\n🟢 Trained parameters loaded from: {}\n'.format(PARAMS_FILE_PATH))
        for theta in thetas:
            print(' '.join(map(str, theta)))
        input('\nPress Enter to continue...\n')
    except FileNotFoundError:
        print('❌ Error: File not found {}'.format(PARAMS_FILE_PATH), file=sys.stderr)
        exit(1)
    except ValueError:
        print('❌ Error: Reading file {}'.format(PARAMS_FILE_PATH), file=sys.stderr)
        exit(1)
    except Exception as e:
        print('❌ Error:', e, file=sys.stderr)
        exit(1)

    # Perform prediction
    predicted_houses = predict(file_path, thetas)

    # Print predictions
    print('\nPredictions:')
    for i, house in enumerate(predicted_houses):
        print(f"Prediction for instance {i+1}: {house}")

    # Save predictions to a CSV file
    save_predictions_to_csv(predicted_houses)


if __name__ == "__main__":
    main()
