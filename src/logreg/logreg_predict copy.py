import os
import sys
import csv

import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(os.path.dirname(current_dir), 'utils')
sys.path.append(module_dir)

from logreg_train import columns_to_drop, houses, PARAMS_FILE_PATH, MyLogisticRegression, normalize_xset
from data_analyzer import percentile

HOUSES_FILE_PATH = 'data/houses.csv'


def predict(filename):
    '''
    Main function to predict the Hogwarts house of students.
    '''
    thetas = []
    try:
        with open(PARAMS_FILE_PATH, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                thetas.append(np.array([float(theta) for theta in row['theta'].split(',')]).reshape(-1, 1))
    except FileNotFoundError:
        print('❌ Error: File not found {}'.format(PARAMS_FILE_PATH),
              file=sys.stderr)
        exit(1)
    except ValueError:
        print('❌ Error: Reading file {}'.format(PARAMS_FILE_PATH),
              file=sys.stderr)
        exit(1)
    except Exception as e:
        print('❌ Error:', e, file=sys.stderr)
        exit(1)

    # models = []
    # for i in range(4):
    #     models.append(MyLogisticRegression(thetas[i]))

    models = [MyLogisticRegression(theta) for theta in thetas]

    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(filename)
        df.info()
        print('\nINIT CSV FILE')
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

    df_num = df.select_dtypes(include=['int', 'float']).copy()

    # Replace NaN data with mean value
    for column in df_num.columns:
        # if df_num[column].dtype != 'object':
        median = percentile(df_num[column], 0.50)
        df_num[column] = df_num[column].fillna(median)

    # df_num.insert(1, 'Hogwarts House', df['Hogwarts House'])

    df_num.drop(columns_to_drop, inplace=True, axis=1)

    df_num.info()
    input('\nPress Enter to continue...\n')

    df_num.drop('Hogwarts House', inplace=True, axis=1)

    df_num = df_num.iloc[:, 1:]

    # Normalizar los datos y obtener las características X
    x = np.array(df_num)
    X_norm, _, _ = normalize_xset(x)

    # # Realizar las predicciones utilizando los modelos
    # predictions = np.zeros((x.shape[0], len(houses)))
    # for model in models:
    #     predictions += model.predict(X_norm)

    # # Encontrar la casa de Hogwarts más probable para cada estudiante
    # final_predictions = np.argmax(predictions, axis=1)

    predictions = np.zeros((x.shape[0], len(houses)))
    for model in models:
        # Realizar la predicción y convertirla en una matriz de columna
        prediction = model.predict(X_norm).reshape(-1, 1)
        predictions += prediction

    final_predictions = np.argmax(predictions, axis=1)

    # df_num = df_num.iloc[:, 1:]

    # df_num.info()

    # input('\nPress Enter to continue...\n')

    # nb_features = len(df_num.columns)
    # print(f'❗️ nb_features: {nb_features}')

    # # set X and y
    # # x = np.array(df_num).reshape(400, nb_features)
    # x = np.array(df_num)
    # X_norm, _, _ = normalize_xset(x)

    # predictions = np.zeros((x.shape[0], len(houses)))
    # for model in models:
    #     predictions += model.predict(X_norm)

    # # Find the most probable house for each student
    # final_predictions = np.argmax(predictions, axis=1)







    # predictions = np.zeros((x.shape[0], len(houses)))
    # for model_idx, model in enumerate(models):
    #     model_predictions = model.predict(X_norm)
    #     predictions[:, model_idx] = model_predictions.flatten()

    # final_predictions = np.argmax(predictions, axis=1)

    # predictions = np.zeros((x.shape[0], len(houses)))
    # for model_idx, model in enumerate(models):
    #     model_predictions = model.predict(X_norm)
    #     predictions += model_predictions

    # final_predictions = np.argmax(predictions, axis=1)



    # predictions = np.zeros((x.shape[0], len(houses)))
    # for model_idx, model in enumerate(models):
    #     model_predictions = model.predict(X_norm)
    #     predictions += model_predictions

    # final_predictions = np.argmax(predictions, axis=1)

    # # Convertir los índices de las predicciones finales en nombres de casas
    # final_house_predictions = [houses[idx] for idx in final_predictions]



    # predict = np.empty((X.shape[0], 0))
    # for model in models:
    #     predict = np.c_[predict, model.predict(X_norm)]
    # predict = np.argmax(predict, axis=1).reshape((-1, 1))

    # Save the predictions to a CSV file
    try:
        with open(HOUSES_FILE_PATH, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(["Index", "Hogwarts House"])
            for index, prediction in enumerate(final_predictions):
                writer.writerow([index, houses[int(prediction)]])
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

    predict(file_path)


if __name__ == "__main__":
    main()
