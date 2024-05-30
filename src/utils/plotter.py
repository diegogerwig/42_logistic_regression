import os
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Unable to import Axes3D")
import matplotlib.pyplot as plt


def plot_loss_history(houses, loss_histories, PLOTS_DIR):
    plt.figure(figsize=(12, 9))
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


def plot_feature_importance(thetas, column_names, houses, PLOTS_DIR):
    num_houses = len(thetas)
    num_features = len(column_names)
    fig, axes = plt.subplots(num_houses, 1, figsize=(12, 9), sharex=True)
    for i, theta in enumerate(thetas):
        feature_importance = abs(theta[1:])
        bars = axes[i].barh(column_names, feature_importance)
        axes[i].set_title(f'House {houses[i]} Feature Importance')
        axes[i].set_xlabel('Importance')
        axes[i].set_ylabel('Feature')
        axes[i].invert_xaxis()
        axes[i].grid(axis='x')
        for bar in bars:
            width = bar.get_width()
            axes[i].text(width, bar.get_y() + bar.get_height()/2,
                         '{:.2f}'.format(width),
                         va='center', ha='left', fontsize=8)
    plt.tight_layout()
    plt.show(block=False)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    save_path = os.path.join(PLOTS_DIR, 'plot_feature_importance.png')
    plt.savefig(save_path)
    print('\n⚪️ Plot saved as: {}\n'.format(save_path))
    input('\nPress Enter to continue...\n')
    plt.close()
