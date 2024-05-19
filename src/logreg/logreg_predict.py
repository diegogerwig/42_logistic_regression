import matplotlib
matplotlib.use('Agg')



import os
import sys
import csv
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(os.path.dirname(current_dir), 'utils')
sys.path.append(module_dir)
from logreg_train import COLUMNS_TO_DROP, HOUSES, PARAMS_FILE_PATH, PLOTS_DIR
from stats import percentile
from normalize import normalize_xset
from gradient import sigmoid
from plotter import plot_feature_importance

HOUSES_FILE_PATH = 'data/houses.csv'


def custom_input(prompt):
    if "--skip-input" in sys.argv:
        return " "
    else:
        return input(prompt)


def load_trained_parameters(PARAMS_FILE_PATH):
    print('\n🔆 LOAD TRAINED PARAMETERS')
    try:
        with open(PARAMS_FILE_PATH, 'r') as file:
            reader = csv.reader(file)
            thetas = [np.array([float(val) for val in row[0].split(',')]) for row in reader]
        print('\n🟢 Trained parameters loaded from: {}\n'.format(PARAMS_FILE_PATH))
        for theta in thetas:
            print(' '.join(map(str, theta)))
        custom_input('\nPress ENTER to continue...\n')
        return thetas
    except FileNotFoundError:
        print('❌ Error: File not found {}'.format(PARAMS_FILE_PATH), file=sys.stderr)
        exit(1)
    except ValueError:
        print('❌ Error: Reading file {}'.format(PARAMS_FILE_PATH), file=sys.stderr)
        exit(1)
    except Exception as e:
        print('❌ Error:', e, file=sys.stderr)
        exit(1)


def check_format(filename, col_names, col_types):
    try:
        print('\n🔆 READ CSV FILE')
        df = pd.read_csv(filename)
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

    col_check = zip(col_names, col_types)

    # Check that the expected columns are present and have the correct type
    if not set(col_names).issubset(df.columns):
        print(f"Missing columns in '{filename}' file", file=sys.stderr)
        exit(1)
    for name, type_ in col_check:
        if not df[name].dtype == type_:
            print(f"Wrong column type in '{filename} file", file=sys.stderr)
            exit(1)


def predict(filename, thetas):
    try:
        # Read the CSV file into a DataFrame
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

    print('\n🔆 REMOVE SOME CATEGORY FEATURES')
    df_num.drop(COLUMNS_TO_DROP, inplace=True, axis=1)
    print(f'   COLUMNS DROPPED: {COLUMNS_TO_DROP}')
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

    print('\n🔆 NORMALIZE DATA')
    x = np.array(df_num_excl_first_two)
    X_norm = normalize_xset(x)
    # Add bias term to normalized features
    X = np.hstack((np.ones((X_norm.shape[0], 1)), X_norm))

    print('\n🔆 PERFORM PREDICTIONS')
    predictions = []
    for theta in thetas:
        probability = sigmoid(np.dot(X, theta))
        predictions.append(probability)
    print('\nPredictions values:')
    np.set_printoptions(suppress=True, threshold=np.inf)
    stacked_predictions = np.column_stack(predictions)
    for pred in predictions:
        print(len(pred))
    print(stacked_predictions)
    custom_input('\nPress ENTER to continue...\n')

    predicted_house_indices = np.argmax(stacked_predictions, axis=1)
    predicted_houses = []
    for idx in predicted_house_indices:
        predicted_houses.append(HOUSES[idx])

    print('\nPredicted houses:')
    print(predicted_houses)

    print('\n🔆 PLOT FEATURE IMPORTANCE')
    skip_input = len(sys.argv) == 3 and sys.argv[2] == "--skip-input"
    if not skip_input:
        plot_feature_importance(thetas, column_names, HOUSES, PLOTS_DIR)

    return predicted_houses


def save_predictions(final_predictions):
    print('\n🔆 SAVING PREDICTIONS')
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

    # Load trained parameters
    thetas = load_trained_parameters(PARAMS_FILE_PATH)

    col_names = ["Index", "Hogwarts House", "First Name", "Last Name",
                 "Birthday", "Best Hand", "Arithmancy", "Astronomy",
                 "Herbology", "Defense Against the Dark Arts",
                 "Divination", "Muggle Studies", "Ancient Runes",
                 "History of Magic", "Transfiguration", "Potions",
                 "Care of Magical Creatures", "Charms", "Flying"]
    col_types = [int, None, object, object, object, object, float, float,
                 float, float, float, float, float, float, float, float, float,
                 float, float]

    # Check format of the CSV file
    check_format(file_path, col_names, col_types)

    predicted_houses = predict(file_path, thetas)

    save_predictions(predicted_houses)


if __name__ == "__main__":
    main()
