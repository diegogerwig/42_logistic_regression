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
# columns_to_drop = ['Defense Against the Dark Arts', 'Arithmancy', 'Care of Magical Creatures']
# columns_to_drop = ['Defense Against the Dark Arts', 'Arithmancy', 'Care of Magical Creatures', 'Transfiguration', 'Charms', 'Flying']
houses = ['Gryffindor', 'Slytherin', 'Hufflepuff', 'Ravenclaw']


class MyLogisticRegression:
    def __init__(self, theta, lr=LEARNING_RATE, max_iter=MAX_ITERATIONS):
        self.theta = theta
        self.lr = lr
        self.max_iter = max_iter
        self.loss_history = {house: [] for house in houses}

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def loss(self, h, y):
        return -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
    
    def fit(self, X, y, model_name, progress_bar=None):
        m = len(y)
        X = np.hstack((np.ones((m, 1)), X))
        for _ in tqdm(range(self.max_iter), desc=f'Training {model_name :<11}'):
            h = self.sigmoid(X.dot(self.theta))
            error = h - y
            gradient = X.T.dot(error) / m
            self.theta -= self.lr * gradient
            current_loss = self.loss(h, y)
            for house in houses:
                self.loss_history[house].append(current_loss)
            if progress_bar:
                progress_bar.update(1)

    def plot_loss_history(self, model_name):
        plt.figure(figsize=(15, 10))
        line_styles = ['-', '--', '-.', ':']
        for i, house in enumerate(houses):
            house_loss_history = np.array(self.loss_history[house])
            line_style = line_styles[i % len(line_styles)]
            plt.plot(house_loss_history, label=house, linestyle=line_style)
        plt.title(f'Loss Function History {model_name}')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)
        os.makedirs(PLOTS_DIR, exist_ok=True)
        save_path = os.path.join(PLOTS_DIR, f'plot_LOSS_history_{model_name}.png')
        plt.savefig(save_path)
        print('\n⚪️ Plot saved as: {}\n'.format(save_path))
        input('\nPress Enter to continue...\n')
        plt.close()
    
    def predict(self, X):
        m = X.shape[0]
        X = np.hstack((np.ones((m, 1)), X))
        probabilities = self.sigmoid(X.dot(self.theta))
        return (probabilities >= 0.5).astype(int)


def normalize_xset(X):
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    X_norm = (X - means) / stds
    return X_norm, means, stds


def train(filename):
    '''
    Main function to train the logistic model.
    '''
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

    # Get numeric features
    df_num = df.select_dtypes(include=['int', 'float'])

    # Replace NaN data with mean value
    for column in df_num.columns:
        # if df_num[column].dtype != 'object':
        median = percentile(df_num[column], 0.50)
        df_num[column] = df_num[column].fillna(median)

    df_num.info()
    print('\nREPLACE NaN DATA WITH MEAN VALUE')
    input('\nPress Enter to continue...\n')

    # Drop category features
    df_num.drop(columns_to_drop, inplace=True, axis=1)

    df_num.info()
    df_num.to_csv('data/df_num.csv', index=False)
    print('\nREMOVE SOME CATEGORY FEATURES')
    input('\nPress Enter to continue...\n')

    ft_describe('data/df_num.csv')

    # Get numeric features
    nb_features = len(df_num.columns)
    x = np.array(df_num)
    y = np.array(df['Hogwarts House'])

    # Normalize data
    X_norm, means, stds = normalize_xset(x)

    # Create label sets to train models
    y_trains = []
    for house in houses:
        y_train_house = np.where(y == house, 1, 0)
        y_trains.append(y_train_house)

    # Train models
    models = []
    for y_train, house in zip(y_trains, houses):
        # Initialize theta with random values
        theta = np.random.rand(nb_features + 1, 1600)
        
        # Create an instance of MyLogisticRegression with the initialized theta
        model = MyLogisticRegression(theta, lr=LEARNING_RATE, max_iter=MAX_ITERATIONS)
        
        # Initialize tqdm to track progress
        progress_bar = tqdm(total=MAX_ITERATIONS, desc=f'Training {house}', disable=True)
        
        # Fit the model with progress bar
        model.fit(X_norm, y_train, house, progress_bar=progress_bar)

        # Close progress bar
        progress_bar.close()

        # Append the trained model to the list of models
        models.append(model)

    model.plot_loss_history('All Houses')

    # Saving hyperparameters adjusted with the training
    try:
        with open(PARAMS_FILE_PATH, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(["theta", "mean", "std"])
            for model, house in zip(models, houses):
                last_theta = model.theta[:, -1]
                theta_str = ','.join([f'{val}' for val in last_theta])
                mean_str = ','.join([f'{mean}' for mean in means])
                std_str = ','.join([f'{std}' for std in stds])
                writer.writerow([theta_str, mean_str, std_str])
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

    # Describe the CSV file
    train(file_path)


if __name__ == "__main__":
    main()
