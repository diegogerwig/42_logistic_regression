import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


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
        df = df.dropna()
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

    plt.show(block=False)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    save_path = os.path.join(PLOTS_DIR, 'pair_plot.png')
    plt.savefig(save_path)
    print('\n⚪️ Plot saved as: {}\n'.format(save_path))
    input('\nPress Enter to continue...\n')
    plt.close()

    df_num = df.select_dtypes(include=['number'])
    corr_matrix = df_num.corr()
    colors = [(0, 'green'), (0.5, 'orange'), (1, 'green')]
    cmap = LinearSegmentedColormap.from_list('Custom', colors)  
    plt.figure(figsize=(15, 12))
    plt.imshow(corr_matrix, cmap=cmap, interpolation='nearest', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Correlation Matrix')
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            plt.text(j, i, "{:.4f}".format(corr_matrix.iloc[i, j]), ha='center', va='center', color='black')
    plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix)), corr_matrix.columns)
    plt.tight_layout()
    plt.show(block=False)

    save_path = os.path.join(PLOTS_DIR, 'correlation_matrix.png')
    plt.savefig(save_path, dpi=300)
    print('\n⚪️ Plot saved as: {}\n'.format(save_path))
    input('\nPress Enter to continue...\n')


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
