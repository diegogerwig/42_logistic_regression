import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
# import time


PLOTS_DIR = './plots'
os.makedirs(PLOTS_DIR, exist_ok=True)


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

    # # Draw pairplot
    # column_combinations = [(col1, col2) for col1 in df.columns for col2 in df.columns if col1 != col2]

    # # Calcular total de combinaciones para la barra de progreso
    # total_combinations = len(column_combinations)

    # # Inicializar la barra de progreso
    # pbar = tqdm(total=total_combinations, desc="Generating P Plot")

    # # Generar pair plot para cada combinación de columnas
    # for combination in column_combinations:
    #     sns.pairplot(df, hue='Hogwarts House', kind='scatter', diag_kind='hist', plot_kws={"s": 4},
    #                  height=0.95, aspect=1.5, x_vars=[combination[0]], y_vars=[combination[1]])
    #     plt.close('all')
    #     pbar.update(1)  
    #     time.sleep(0.001)  

    # pbar.close()  

    with tqdm(total=1, desc="Generando Pair Plot") as pbar:
        sns.pairplot(df, hue='Hogwarts House', kind='scatter', diag_kind='hist',
                     plot_kws={"s": 4}, height=0.95, aspect=1.5)
        pbar.update(1)
        plt.show(block=False)
        save_path = os.path.join(PLOTS_DIR, 'pair_plot.png')
        plt.savefig(save_path)
        print('\n⚪️ Plot saved as: {}\n'.format(save_path))
        input('\nPress Enter to continue...\n')
        plt.close()


    # sns.pairplot(df, hue='Hogwarts House', kind='scatter', diag_kind='hist',
    #              plot_kws={"s": 4}, height=0.95, aspect=1.5)

    # plt.show(block=False)

    # save_path = os.path.join(PLOTS_DIR, 'pair_plot.png')
    # plt.savefig(save_path)
    # print('\n⚪️ Plot saved as: {}\n'.format(save_path))
    # input('\nPress Enter to continue...\n')
    # plt.close()

    df_num = df.select_dtypes(include=['number'])
    corr_matrix = df_num.corr()
    # colors = [(0, 'lightblue'), (0.5, 'white'), (1, 'lightgreen')]
    # cmap = LinearSegmentedColormap.from_list('Custom', colors)  
    plt.figure(figsize=(15, 12))
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)
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
