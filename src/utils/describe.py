import sys
import os
import pandas as pd
from data_analyzer import mean, std, percentile, mode, frequency


def data_stats(feature):
    '''
    Analyzes a feature (column) of a DataFrame.
    Computes basic descriptive statistics for the given feature.
    '''
    # Remove NaN values and sort the feature
    feature = feature.dropna().sort_values()
    # Calculate descriptive statistics
    return {
        'name': feature.name,
        'count': feature.size,
        'mean': mean(feature),
        'std': std(feature, mean(feature)),
        'min': percentile(feature, 0),
        '25%': percentile(feature, 0.25),
        '50%': percentile(feature, 0.50),
        '75%': percentile(feature, 0.75),
        'max': percentile(feature, 1),
        '-----': 0,
        'mode': mode(feature),
        'freq': frequency(feature)
    }


def display(data):
    '''
    Displays descriptive statistics of features in blocks.
    '''
    SIZE = 7
    print()

    for i in range(0, len(data), SIZE):
        slce = data[i:i + SIZE]

        # Display header
        line = ' ' * 6
        for desc in slce:
            line += f'{desc["name"][:11]:>13s}...' if len(desc["name"]) \
                > 14 else f'{desc["name"]:>16s}'
        print(line)

        # Display statistics
        for legend in ['count', 'mean', 'std', 'min',
                       '25%', '50%', '75%', 'max', '-----', 'mode', 'freq']:
            line = f'{legend:6s}'
            if legend == '-----':  # Check if legend is '----'
                line += '-' * 112  # Print dashes for each column
            else:
                for desc in slce:
                    line += f'{desc[legend]:16.6f}' \
                        if -1000000 <= desc[legend] <= 1000000 \
                        else f'{desc[legend]:16.6e}'
            print(line)

        print()

    input('\nPress Enter to continue...\n')


def ft_describe(filename):
    '''
    Describes the numeric features of a CSV file.
    '''
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(filename)
        df.info()
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

    # Analyze numeric features of the DataFrame
    stats = [data_stats(df[col]) for col in df if
             pd.api.types.is_numeric_dtype(df[col])]

    display(stats)

    # Ask user if they want to replace null values with median
    # Prompt the user for input
    if df.isnull().values.any():
        response = input('❗️ Do you want to replace null values with median? (yes/no): ').lower()

        if response == 'yes':
            # Replace null values with median
            for column in df.columns:
                if df[column].dtype != 'object':
                    median = percentile(df[column], 0.50)
                    df[column] = df[column].fillna(median)
            df.info()
            # Export the modified DataFrame to a new CSV file
            new_filename = filename.replace('.csv', '_clean.csv')
            df.to_csv(new_filename, index=False)
            print(f'\n✅ Data exported to "{new_filename}"')

            input('\nPress Enter to continue...\n')

            # Analyze numeric features of the DataFrame
            stats = [data_stats(df[col]) for col in df if
                    pd.api.types.is_numeric_dtype(df[col])]
            # Display statistics
            display(stats)


def main():
    '''
    Main function of the program.
    '''
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 2:
        print('❗️ Usage: python3 describe.py data*.csv')
        exit(1)

    # Get the file path from command line arguments
    file_path = sys.argv[1]
    print(f'\n🟡 File path:{file_path}\n')

    # Check if the file exists
    if not os.path.isfile(file_path):
        print('❌ Error: File not found')
        exit(1)

    # Describe the CSV file
    describe(file_path)


if __name__ == '__main__':
    main()
