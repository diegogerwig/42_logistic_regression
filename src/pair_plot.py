import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


PLOTS_DIR = './plots'

categories = ['Hogwarts House', 'Arithmancy', 'Astronomy', 'Herbology',
              'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
              'Ancient Runes', 'History of Magic', 'Transfiguration',
              'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']


def pair_plot(filename):
    '''
    Generate pair plot for the given dataset file.
    '''
    # Read data
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print('❌ Error: File not found.')
        exit(1)
    except pd.errors.EmptyDataError:
        print('❌ Error: Dataset file is empty.')
        exit(1)
    except pd.errors.ParserError:
        print('❌ Error: Invalid CSV file format.')
        exit(1)
    except Exception as e:
        print('❌ Error:', e)
        exit(1)

    # Select features
    df = df[categories]

    # Set the font size of axis labels
    sns.set_context("paper", font_scale=0.6)

    # Draw pairplot
    sns.pairplot(df, hue='Hogwarts House', kind='scatter', diag_kind='hist',
                 plot_kws={"s": 4}, height=0.95, aspect=1.5)

    # Show plot
    plt.show(block=False)

    # Save plot
    os.makedirs(PLOTS_DIR, exist_ok=True)
    save_path = os.path.join(PLOTS_DIR, 'pair_plot.png')
    plt.savefig(save_path)
    print('\n⚪️ Plot saved as: {}\n'.format(save_path))
    input('\nPress Enter to continue...\n')
    plt.close()


def main():
    '''
    Main function of the program.
    '''
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 2:
        print('❗️ Usage: python3 scatter_plot.py data*.csv')
        exit(1)

    # Get the file path from command line arguments
    file_path = sys.argv[1]
    print(f'\n🟡 File path:{file_path}\n')

    # Check if the file exists
    if not os.path.isfile(file_path):
        print('❌ Error: File not found')
        exit(1)

    # Plot the scatter plot
    pair_plot(file_path)


if __name__ == "__main__":
    main()
