import sys
import os
import csv

import pandas as pd
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(os.path.dirname(current_dir), 'utils')
sys.path.append(module_dir)
from stats import percentile
from normalize import normalize_xset
from gradient import gradient_descent
from accuracy import accuracy
from plotter import plot_loss_history

MAX_ITERATIONS = 10000
LEARNING_RATE = 0.01
PARAMS_FILE_PATH = 'data/params.csv'
PLOTS_DIR = './plots'

columns_to_drop = []
# columns_to_drop = ['Astronomy', 'History of Magic', 'Transfiguration', 'Charms', 'Flying']
# columns_to_drop = ['Astronomy', 'History of Magic', 'Transfiguration', 'Charms', 'Flying', 'Arithmancy', 'Care of Magical Creatures']
# columns_to_drop = ['Astronomy', 'History of Magic', 'Transfiguration', 'Charms', 'Flying', 'Arithmancy', 'Care of Magical Creatures', 'Herbology', 'Potions', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes']

houses = ['Gryffindor', 'Slytherin', 'Hufflepuff', 'Ravenclaw']


def custom_input(prompt):
    if "--skip-input" in sys.argv:
        return " "
    else:
        return input(prompt)


def save_parameters(thetas, PARAMS_FILE_PATH):
    print('\n🔆 SAVING PARAMETERS')
    try:
        with open(PARAMS_FILE_PATH, 'w', newline='') as file:
            writer = csv.writer(file)
            for theta in thetas:
                theta_str = ','.join([str(val) for val in theta.flatten()])
                writer.writerow([theta_str])
        print('\n⚪️ Parameters file saved as: {}\n'.format(PARAMS_FILE_PATH))
    except FileNotFoundError:
        print('❌ Error: File not found {}'.format(PARAMS_FILE_PATH),
              file=sys.stderr)
        exit(1)
    except Exception as e:
        print('❌ Error:', e, file=sys.stderr)
        exit(1)


def train(filename):
    '''
    Main function to train the logistic model.
    '''
    try:
        print('\n🔆 INIT CSV FILE')
        df = pd.read_csv(filename)
        print(f'\n🟢 File "{filename}" loaded successfully\n')
        df.info()
        print(df)
        custom_input('\nPress ENTER to continue...\n')
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

    print('\n🔆 GET NUMERIC FEATURES')
    df_num = df.select_dtypes(include=['int', 'float']).copy()
    df_num.info()
    print(df_num)
    custom_input('\nPress ENTER to continue...\n')

    print('\n🔆 REPLACE NaN DATA WITH MEDIAN VALUE')
    for column in df_num.columns:
        median = percentile(df_num[column], 0.50)
        df_num[column] = df_num[column].fillna(median)
    df_num.info()
    print(df_num)
    custom_input('\nPress ENTER to continue...\n')
    
    print('\n🔆 INSERT HOGWARTS HOUSE COLUMN')
    df_num.insert(1, 'Hogwarts House', df['Hogwarts House'])
    df_num.info()
    print(df_num)
    custom_input('\nPress ENTER to continue...\n')

    print('\n🔆 REMOVE SOME CATEGORY FEATURES')
    df_num.drop(columns_to_drop, inplace=True, axis=1)
    print(f'   COLUMNS DROPPED: {columns_to_drop}')
    df_num.info()
    print(df_num)
    custom_input('\nPress ENTER to continue...\n')

    print('\n🔆 REMOVE FIRST TWO COLUMNS')
    df_num_excl_first_two = df_num.iloc[:, 2:]
    df_num_excl_first_two.info()
    print(df_num_excl_first_two)
    custom_input('\nPress ENTER to continue...\n')

    nb_features = len(df_num_excl_first_two.columns)
    column_names = df_num_excl_first_two.columns.tolist()
    print(f'\n🔆 NUMBER OF FEATURES: {nb_features}')
    print(column_names)
    custom_input('\nPress ENTER to continue...\n')

    print('\n🔆 CONVERT DATAFRAME TO NUMPY ARRAY')
    x = np.array(df_num_excl_first_two)
    print(x.shape)
    print(x)
    custom_input('\nPress ENTER to continue...\n')

    print('\n🔆 GET HOGWARTS HOUSE COLUMN')
    y = np.array(df_num['Hogwarts House'])
    print(y.shape)
    print(y)
    custom_input('\nPress ENTER to continue...\n')

    print('\n🔆 NORMALIZE DATA')
    X_norm = normalize_xset(x)
    print(X_norm.shape)
    print(X_norm)
    custom_input('\nPress ENTER to continue...\n')

    print('\n🔆 CREATE LABEL SETS')
    y_trains = []
    for house in houses:
        y_train_house = np.where(y == house, 1, 0)
        y_trains.append(y_train_house)
    for i, house in enumerate(houses):
        print(f"Labels for {house}: \t{y_trains[i][:20]}")
    custom_input('\nPress ENTER to continue...\n')

    print('\n🔆 INITIALIZE THETAS')
    thetas = []
    for _ in range(4):
        theta = np.zeros((nb_features + 1, 1))
        thetas.append(theta)
    for i, theta in enumerate(thetas):
        print(f"   Parameters for house {houses[i]} (shape {theta.shape}): \n{theta}")
    custom_input('\nPress ENTER to continue...\n')

    print('\n🔆 TRAINING')
    loss_histories = []
    for i, house in enumerate(houses):
        print(f"\n🟡 Training for house: {house}")
        # Add bias term to normalized features
        X = np.hstack((np.ones((X_norm.shape[0], 1)), X_norm))
        theta, J_history = gradient_descent(X, y_trains[i].reshape(-1, 1), thetas[i], LEARNING_RATE, MAX_ITERATIONS)
        thetas[i] = theta
        loss_histories.append(J_history)
    custom_input('\nPress ENTER to continue...\n')

    print('\n🔆 PLOTTING LOSS HISTORY')
    skip_input = len(sys.argv) == 3 and sys.argv[2] == "--skip-input"
    if not skip_input:
        plot_loss_history(houses, loss_histories, PLOTS_DIR)

    print('\n🔆 CALCULATING ACCURACY')
    accuracies = []
    for i, house in enumerate(houses):
        # print(f"Evaluating accuracy for house: {house}")
        X = np.hstack((np.ones((X_norm.shape[0], 1)), X_norm))
        acc = accuracy(X, y_trains[i].reshape(-1, 1), thetas[i])
        accuracies.append(acc)
        print(f"\nAccuracy for {house}: {acc * 100:.4f}%")

    mean_accuracy = np.mean(accuracies) * 100
    if mean_accuracy >= 99:
        print(f"\n✅ Mean accuracy across all houses: {mean_accuracy:.4f}%.")
    else:
        print(f"\n❌ Mean accuracy across all houses: {mean_accuracy:.4f}%")
    custom_input('\nPress ENTER to continue...\n')

    save_parameters(thetas, PARAMS_FILE_PATH)


def main():
    '''
    Main function of the program.
    '''
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print('❗️ Usage: python3 script.py data*.csv [--skip-input]')
        exit(1)
                
    if len(sys.argv) == 3 and sys.argv[2] != '--skip-input':
        print('❌ Error: Invalid argument') 
        exit(1)

    # Get the file path from command line arguments
    file_path = sys.argv[1]
    print(f'\n🟡 File path:{file_path}\n')

    # Check if the file exists
    if not os.path.isfile(file_path):
        print('❌ Error: File not found')
        exit(1)

    train(file_path)


if __name__ == "__main__":
    main()
