import sys
import os
import pandas as pd
import matplotlib.pyplot as plt


PLOTS_DIR = './plots'

categories = ['Arithmancy', 'Astronomy', 'Herbology',
              'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
              'Ancient Runes', 'History of Magic', 'Transfiguration',
              'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']

houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']

colors = ['blue', 'orange', 'green', 'red']


def scatter_plot(filename):
    '''
    Generate scatter plot for the given dataset file.
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

    # Create plot grid
    _, axes = plt.subplots(nrows=len(categories), ncols=len(categories),
                           figsize=(20, 12))

    # Adjust spacing between subplots
    plt.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.05,
                        hspace=0.1, wspace=0.1)

    # Turn grid into array
    axes = axes.flatten()

    # Compare each class to each class
    for i, class_x in enumerate(categories):
        for j, class_y in enumerate(categories):
            # Select the right subplot
            ax = axes[i * len(categories) + j]

            # Set column titles
            if i == 0:
                # Alternate placement for readability
                if j % 2 == 0:
                    ax.set_title(class_y, fontsize=7)
                else:
                    plt.text(0.5, 1.5, class_y, horizontalalignment='center',
                             fontsize=7, transform=ax.transAxes)

            # Set line titles
            if j == 0:
                # Alternate placement for readability
                if i % 2 == 0:
                    ax.set_ylabel(class_x, rotation=90, fontsize=7)
                else:
                    plt.text(-0.3, 0.5, class_x, rotation=90,
                             verticalalignment='center', fontsize=7,
                             transform=ax.transAxes)

            # Draw scatter plot for each house
            for house, color in zip(houses, colors):
                # Filter data for current house
                house_data = df[df['Hogwarts House'] == house]
                ax.scatter(house_data[class_x], house_data[class_y],
                           s=4, edgecolor='grey', linewidth=0.5,
                           alpha=0.75, color=color)

            ax.tick_params(left=False, right=False, labelleft=False,
                           labelbottom=False, bottom=False)

    # Add legend
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w',
                                 markersize=5, markerfacecolor=color,
                                 label=house)
                      for house, color in zip(houses, colors)]
    plt.legend(handles=legend_handles, labels=houses, loc='lower center',
               bbox_to_anchor=(-1.0, -0.5),
               ncol=len(houses), fontsize=7)

    # Show plot
    plt.show(block=False)

    # Save plot
    os.makedirs(PLOTS_DIR, exist_ok=True)
    save_path = os.path.join(PLOTS_DIR, 'scatter_plot.png')
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
    scatter_plot(file_path)


if __name__ == "__main__":
    main()
