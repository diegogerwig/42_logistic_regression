import os
import numpy as np
import matplotlib.pyplot as plt


def plot_loss_history(houses, loss_histories, PLOTS_DIR):
    plt.figure(figsize=(15, 10))
    line_styles = ['-']
    for i, house in enumerate(houses):
        house_loss_history = np.array(loss_histories[i])
        line_style = line_styles[i % len(line_styles)]
        plt.plot(house_loss_history, label=house, linestyle=line_style)
    plt.title('Loss Function History')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    save_path = os.path.join(PLOTS_DIR, 'plot_loss_history.png')
    plt.savefig(save_path)
    print('\n⚪️ Plot saved as: {}\n'.format(save_path))
    input('\nPress Enter to continue...\n')
    plt.close()