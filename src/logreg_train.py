import sys
import os
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def train(filename):
    '''
    Train the logistic regression model.
    '''
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(filename)
        df = df.dropna()
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
