import os
import sys
import csv

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

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
    means = []
    stds = []
    try:
        with open(PARAMS_FILE_PATH, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                thetas.append(np.array([float(theta) for theta in row['theta'].split(',')]).reshape(-1, 1))
                means.append(np.array([float(mean) for mean in row['mean'].split(',')]).reshape(-1, 1))
                stds.append(np.array([float(std) for std in row['std'].split(',')]).reshape(-1, 1))


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

    # 2. create 4 models with the proper thetas, one per class to classify
    models = []
    for i in range(4):
        models.append(MyLogisticRegression(thetas[i]))



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

 
    df_num = df.select_dtypes(include=['int', 'float'])

 

    # Replace NaN data with mean value
    for column in df_num.columns:
        # if df_num[column].dtype != 'object':
        median = percentile(df_num[column], 0.50)
        df_num[column] = df_num[column].fillna(median)

    df_num.drop(columns_to_drop, inplace=True, axis=1)

    df_num.info()
    input('\nPress Enter to continue...\n')

    df_num.drop('Hogwarts House', inplace=True, axis=1)

    df_num.info()
    input('\nPress Enter to continue...\n')

    # nb features
    nb_features = len(df_num.columns)

    # set X and y
    X = np.array(df_num).reshape(-1, nb_features)
    X_norm, _, _ = normalize_xset(X)

    predict = np.empty((X.shape[0], 0))
    for model in models:
        predict = np.c_[predict, model.predict(X_norm)]
    predict = np.argmax(predict, axis=1).reshape((-1, 1))

    try:
        with open(HOUSES_FILE_PATH, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(["Index", "Hogwarts House"])
            for index, prediction in enumerate(predict):
                writer.writerow([index, houses[int(prediction.item())]])
        # print("\033[32mPrediction file has been created: "
        #       "data/houses.csv\033[0m")
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

    # Describe the CSV file
    predict(file_path)


if __name__ == "__main__":
    main()


