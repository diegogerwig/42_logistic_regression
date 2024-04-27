import sys
import os
import pandas as pd


def correlation(filename):
    try:
        print('\n🟡 INIT CSV FILE')
        data = pd.read_csv(filename)
        print(f'🟢 File "{filename}" loaded successfully\n')

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

    variables = data[['Astronomy', 'History of Magic', 'Transfiguration', 'Charms', 'Flying',
                    'Arithmancy', 'Care of Magical Creatures', 'Herbology', 'Potions',
                    'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes']]

    correlation_matrix = variables.corr()

    pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))

    results_df = pd.DataFrame(pairs, columns=['Variable 1', 'Variable 2', 'Coeficiente de Correlación'])

    results_df['Abs_Correlation'] = results_df['Coeficiente de Correlación'].abs()
    results_df = results_df.sort_values(by='Abs_Correlation', ascending=False).drop(columns=['Abs_Correlation'])
  
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    print(results_df)


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
