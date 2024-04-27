import sys
import os
import csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(os.path.dirname(current_dir), 'utils')
sys.path.append(module_dir)
from describe import ft_describe
from data_analyzer import percentile

MAX_ITERATIONS = 10000
LEARNING_RATE = 0.01
PARAMS_FILE_PATH = 'data/params.csv'
PLOTS_DIR = './plots'

columns_to_drop = []
# columns_to_drop = ['Astronomy', 'History of Magic', 'Transfiguration', 'Charms', 'Flying']
# columns_to_drop = ['Astronomy', 'History of Magic', 'Transfiguration', 'Charms', 'Flying', 'Arithmancy', 'Care of Magical Creatures']
# columns_to_drop = ['Astronomy', 'History of Magic', 'Transfiguration', 'Charms', 'Flying', 'Arithmancy', 'Care of Magical Creatures', 'Herbology', 'Potions', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes']

houses = ['Gryffindor', 'Slytherin', 'Hufflepuff', 'Ravenclaw']


def normalize_xset(x):
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)
    X_norm = (x - means) / stds
    return X_norm, means, stds


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []
    for _ in tqdm(range(num_iters)):
        h = sigmoid(np.dot(X, theta))
        J = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        J_history.append(J)
        gradient = (1 / m) * np.dot(X.T, (h - y))
        theta -= alpha * gradient
    return theta, J_history


def accuracy(X, y, theta):
    m = len(y)
    y_pred = sigmoid(np.dot(X, theta))
    y_pred_class = (y_pred >= 0.5).astype(int)
    accuracy = np.mean(y_pred_class == y)
    return accuracy


def plot_loss_history(houses, loss_histories):
    plt.figure(figsize=(15, 10))
    line_styles = ['-']
    for i, house in enumerate(houses):
        house_loss_history = np.array(loss_histories[i])
        line_style = line_styles[i % len(line_styles)]
        plt.plot(house_loss_history, label=house, linestyle=line_style)
    plt.title('Loss Function History')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    save_path = os.path.join(PLOTS_DIR, 'plot_loss_history.png')
    plt.savefig(save_path)
    print('\n⚪️ Plot saved as: {}\n'.format(save_path))
    input('\nPress Enter to continue...\n')
    plt.close()


def train(filename):
    '''
    Main function to train the logistic model.
    '''
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
    
    df_num.insert(1, 'Hogwarts House', df['Hogwarts House'])
    print('\nINSERT HOGWARTS HOUSE COLUMN')
    df_num.info()
    print(df_num)
    input('\nPress Enter to continue...\n')

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

    x = np.array(df_num_excl_first_two)
    print('\nCONVERT DATAFRAME TO NUMPY ARRAY')
    print(x.shape)
    print(x)
    input('\nPress Enter to continue...\n')

    y = np.array(df_num['Hogwarts House'])
    print('\nGET HOGWARTS HOUSE COLUMN')
    print(y.shape)
    print(y)
    input('\nPress Enter to continue...\n')

    # Normalize data
    X_norm, means, stds = normalize_xset(x)
    print('\nNORMALIZE DATA')
    print(X_norm.shape)
    print(X_norm)
    input('\nPress Enter to continue...\n')

    # Create label sets to train models
    y_trains = []
    for house in houses:
        y_train_house = np.where(y == house, 1, 0)
        y_trains.append(y_train_house)
    print('\nCREATE LABEL SETS')
    for i, house in enumerate(houses):
        print(f"Labels for {house}: \t{y_trains[i][:20]}")
    input('\nPress Enter to continue...\n')

    thetas = []  
    for _ in range(4):  
        theta = np.zeros((nb_features + 1, 1))  
        thetas.append(theta)
    print('\nINITIALIZE THETAS')
    for i, theta in enumerate(thetas):
        print(f"Parameters for house {houses[i]} (shape {theta.shape}): \n{theta}")
    input('\nPress Enter to continue...\n')

    # for i, house in enumerate(houses):
    #     print(f"Training for house: {house}")
    #     X = np.hstack((np.ones((X_norm.shape[0], 1)), X_norm))
    #     theta, J_history = gradient_descent(X, y_trains[i].reshape(-1, 1), thetas[i], LEARNING_RATE, MAX_ITERATIONS)
    #     thetas[i] = theta

    loss_histories = []
    for i, house in enumerate(houses):
        print(f"Training for house: {house}")
        X = np.hstack((np.ones((X_norm.shape[0], 1)), X_norm))
        theta, J_history = gradient_descent(X, y_trains[i].reshape(-1, 1), thetas[i], LEARNING_RATE, MAX_ITERATIONS)
        thetas[i] = theta
        loss_histories.append(J_history)

    accuracies = []
    for i, house in enumerate(houses):
        # print(f"Evaluating accuracy for house: {house}")
        X = np.hstack((np.ones((X_norm.shape[0], 1)), X_norm))
        acc = accuracy(X, y_trains[i].reshape(-1, 1), thetas[i])
        accuracies.append(acc)
        print(f"Accuracy for {house}: {acc:.4f}")

    mean_accuracy = np.mean(accuracies)
    print(f"\nMean accuracy across all houses: {mean_accuracy:.4f}")

    plot_loss_history(houses, loss_histories)

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

    train(file_path)


if __name__ == "__main__":
    main()
