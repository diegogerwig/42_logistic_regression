import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PLOTS_DIR = './plots'
os.makedirs(PLOTS_DIR, exist_ok=True)


categories = ['Arithmancy', 'Astronomy', 'Herbology',
              'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
              'Ancient Runes', 'History of Magic', 'Transfiguration',
              'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']

houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']

colors = ['blue', 'orange', 'green', 'red']


def histogram(filename):
    '''
    Generate histogram for the given dataset file.
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

    # Calculate the number of categories with data
    num_categories_with_data = sum(df[col].notnull().any() for col in categories)

    # Calculate the number of rows and columns for subplots
    num_cols = min(num_categories_with_data, 7)
    num_rows = (num_categories_with_data - 1) // num_cols + 1

    # Create plot grid
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 12))

    # Adjust spacing between subplots
    plt.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.05, hspace=0.8, wspace=0.3)

    # Draw histogram and KDE for each category
    for ax, class_ in zip(axes.flatten(), categories):
        if class_ in df.columns and not df[class_].isnull().all():
            ax.set_title(class_, fontsize=10)
            
            # Draw histogram and KDE for each house
            for house, color in zip(houses, colors):
                sns.histplot(df.loc[df['Hogwarts House'] == house, class_],
                            color=color, alpha=0.5, bins=20, label=house, ax=ax, kde=True)

    # Get legend for the first subplot
    handles, labels = axes[0, 0].get_legend_handles_labels()

    # Create a separate subplot for legend
    ax_legend = plt.subplot(111)
    ax_legend.axis('off')

    # Add legend outside the plot
    ax_legend.legend(handles, labels, loc='lower right')

    # Adjust the layout to make room for the legend
    plt.tight_layout()

    # Show plot
    plt.show(block=False)

    # Save plot
    save_path = os.path.join(PLOTS_DIR, 'histogram_with_kde.png')
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
        print('❗️ Usage: python3 histogram.py data*.csv')
        exit(1)

    # Get the file path from command line arguments
    file_path = sys.argv[1]
    print(f'\n🟡 File path:{file_path}\n')

    # Check if the file exists
    if not os.path.isfile(file_path):
        print('❌ Error: File not found')
        exit(1)

    # Plot the histogram
    histogram(file_path)


if __name__ == '__main__':
    main()
