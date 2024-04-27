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

MAX_ITERATIONS = 100
LEARNING_RATE = 1
PARAMS_FILE_PATH = 'data/params.csv'
PLOTS_DIR = './plots'

columns_to_drop = []
# columns_to_drop = ['Astronomy', 'History of Magic', 'Transfiguration', 'Charms', 'Flying']
# columns_to_drop = ['Astronomy', 'History of Magic', 'Transfiguration', 'Charms', 'Flying', 'Arithmancy', 'Care of Magical Creatures']
# columns_to_drop = ['Astronomy', 'History of Magic', 'Transfiguration', 'Charms', 'Flying', 'Arithmancy', 'Care of Magical Creatures', 'Herbology', 'Potions', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes']

houses = ['Gryffindor', 'Slytherin', 'Hufflepuff', 'Ravenclaw']


# class MyLogisticRegression:
#     def __init__(self, thetas, lr=LEARNING_RATE, max_iter=MAX_ITERATIONS):
#         self.thetas = thetas  # Ahora es una lista de thetas para cada casa
#         self.lr = lr
#         self.max_iter = max_iter
#         self.loss_history = {house: [] for house in houses}

#     def sigmoid(self, z):
#         return 1 / (1 + np.exp(-z))
    
#     def loss(self, h, y):
#         return -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
    
#     def fit(self, X, y, model_name, progress_bar=None):
#         m = len(y)
#         X = np.hstack((np.ones((m, 1)), X))
#         for i, (theta, house) in enumerate(zip(self.thetas, houses)):
#             for _ in tqdm(range(self.max_iter), desc=f'Training {model_name} for {house}'):
#                 print("Shape of X:", X.shape)
#                 print("Shape of theta:", theta.shape)
#                 h = self.sigmoid(X.dot(theta.reshape(-1, 1)))
#                 error = h - y
#                 gradient = X.T.dot(error) / m
#                 updated_theta = theta - self.lr * gradient
#                 self.thetas[i] = updated_theta
#                 print("🟣 Shape of updated theta:", updated_theta.shape)
#                 current_loss = self.loss(h, y)
#                 self.loss_history[house].append(current_loss)
#                 if progress_bar:
#                     progress_bar.update(1)

#     def plot_loss_history(self, model_name):
#         plt.figure(figsize=(15, 10))
#         line_styles = ['-', '--', '-.', ':']
#         for i, house in enumerate(houses):
#             house_loss_history = np.array(self.loss_history[house])
#             line_style = line_styles[i % len(line_styles)]
#             plt.plot(house_loss_history, label=house, linestyle=line_style)
#         plt.title(f'Loss Function History {model_name}')
#         plt.xlabel('Iteration')
#         plt.ylabel('Loss')
#         plt.grid(True)
#         plt.legend()
#         plt.tight_layout()
#         plt.show(block=False)
#         os.makedirs(PLOTS_DIR, exist_ok=True)
#         save_path = os.path.join(PLOTS_DIR, f'plot_LOSS_history_{model_name}.png')
#         plt.savefig(save_path)
#         print('\n⚪️ Plot saved as: {}\n'.format(save_path))
#         input('\nPress Enter to continue...\n')
#         plt.close()
    
#     def predict(self, X):
#         m = X.shape[0]
#         X = np.hstack((np.ones((m, 1)), X))
#         probabilities = []
#         for theta in self.thetas:  # Iterar sobre las thetas
#             probabilities.append(self.sigmoid(X.dot(theta)))
#         probabilities = np.array(probabilities)
#         return np.argmax(probabilities, axis=0)


def normalize_xset(x):
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)
    X_norm = (x - means) / stds
    return X_norm, means, stds


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

    # df_num.to_csv('data/df_num.csv', index=False)
    # print('\nREMOVE SOME CATEGORY FEATURES')
    # print(f'COLUMNS DROPPED: {columns_to_drop}')
    # print(df_num.shape)
    # input('\nPress Enter to continue...\n')

    # ft_describe('data/df_num.csv')
    # print(df_num.shape)


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

    thetas = []  # Aquí se inicializan los thetas para cada casa
    for _ in range(4):  # Para cada casa
        theta = np.zeros((nb_features + 1, 1))  # Inicializar theta como un vector de ceros
        # theta = np.expand_dims(theta, axis=1)  # Expandir la dimensión de theta
        thetas.append(theta)
    print('\nINITIALIZE THETAS')
    for i, theta in enumerate(thetas):
        print(f"Parameters for house {houses[i]} (shape {theta.shape}): \n{theta}")
    input('\nPress Enter to continue...\n')

    # models = []
    # for i, (theta, y_train, house) in enumerate(zip(thetas, y_trains, houses)):
    #     model = MyLogisticRegression(theta, lr=LEARNING_RATE, max_iter=MAX_ITERATIONS)
    #     progress_bar = tqdm(total=MAX_ITERATIONS, desc=f'Training {house}', disable=True)
    #     print("🟢 Shape of X_norm:", X_norm.shape)
    #     print("🟢 Shape of y_train:", y_train.shape)
    #     model.fit(X_norm, y_train, house, progress_bar=progress_bar)
    #     progress_bar.close()
    #     models.append(model)


    # model.plot_loss_history('All Houses')

    # Save the parameters to a CSV file
    # try:
    #     with open(PARAMS_FILE_PATH, 'w') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(["theta"])
    #         for model in models:
    #             last_theta = model.thetas[:, -1]  # Corregir aquí
    #             theta_str = ','.join([f'{val}' for val in last_theta])
    #             writer.writerow([theta_str])
    #     print('\n⚪️ Parameters file saved as: {}\n'.format(PARAMS_FILE_PATH))
    # except FileNotFoundError:
    #     print('❌ Error: File not found {}'.format(PARAMS_FILE_PATH),
    #           file=sys.stderr)
    #     exit(1)
    # except Exception as e:
    #     print('❌ Error:', e, file=sys.stderr)
    #     exit(1)


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
