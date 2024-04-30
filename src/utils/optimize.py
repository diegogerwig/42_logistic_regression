import sys
import os
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(os.path.dirname(current_dir), 'logreg')
sys.path.append(module_dir)
from logreg_train import train


def correlation(filename, variables, removed_features):
    try:
        print('\n🔆 READ CSV FILE')
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

    data.dropna(inplace=True)

    variables = data[variables]

    variables = variables.drop(columns=[var for var in removed_features if var in variables.columns])

    correlation_matrix = variables.corr()

    pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))

    results_df = pd.DataFrame(pairs, columns=['Variable 1', 'Variable 2', 'Correlation coefficient'])

    results_df['Abs_Correlation'] = results_df['Correlation coefficient'].abs()
    results_df = results_df.sort_values(by='Abs_Correlation', ascending=False).drop(columns=['Abs_Correlation'])

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    print(results_df.head(10))

    close_to_1 = results_df[results_df['Correlation coefficient'].abs() > 0.99]

    product_correlation = correlation_matrix.abs().prod()
    variable_max_corr_product = product_correlation.idxmax()

    return variable_max_corr_product, close_to_1


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

    variables = ['Astronomy', 'History of Magic', 'Transfiguration', 'Charms', 'Flying',
                 'Arithmancy', 'Care of Magical Creatures', 'Herbology', 'Potions',
                 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes']
    removed_features = []

    while len(variables) > 1:
        accuracy = train(file_path, removed_features, True)
        print(f"\nMean accuracy across all houses: {accuracy:.4f}%.")
        variables = [var for var in variables if var not in removed_features]
        print(f'\n🟩 Current features ({len(variables)}): {variables}')
        print(f'\n🟥 Removed features ({len(removed_features)}): {removed_features}')
        variable_max_corr_product, close_to_1 = correlation(file_path, variables, removed_features)
        if not close_to_1.empty:
            print(f"\n❗️ Warning: Variables with correlation coefficients close to 1 or -1: {close_to_1[['Variable 1', 'Variable 2']].values.tolist()}\n")
            answer = input(f"Do you want to discard some feature? {' / '.join(close_to_1[['Variable 1', 'Variable 2']].values.flatten())} (v1/v2/no): ").lower()
            if answer == 'v1':
                removed_features.append(close_to_1['Variable 1'].values[0])
                variables.remove(close_to_1['Variable 1'].values[0])
            elif answer == 'v2':
                removed_features.append(close_to_1['Variable 2'].values[0])
                variables.remove(close_to_1['Variable 2'].values[0])
            else:
                break
        else:
            print(f'\n🏆 Feature with highest product of absolute correlation coefficients: {variable_max_corr_product}\n')
            answer = input('Do you want to discard this feature? (yes/no): ').lower()
            if answer == 'yes':
                removed_features.append(variable_max_corr_product)
                variables.remove(variable_max_corr_product)
            else:
                break

    print('\n🔆 FINAL RESULT')
    print(f'\n🟩 Current features ({len(variables)}): {variables}')
    print(f'\n🟥 Removed features ({len(removed_features)}): {removed_features}')


if __name__ == "__main__":
    main()
