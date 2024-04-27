import sys
import os
import csv

import pandas as pd
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(os.path.dirname(current_dir), 'logreg')
sys.path.append(module_dir)
from train import train

def generate_combinations(features):
    """

    """
    combinations = []
    nb_combinations = 2**len(features)
    n = len(features)
    for i in range(1, 2**n):
        combination = [features[j] for j in range(n) if (i & (1 << j))]
        combinations.append(combination)
    return combinations


def find_best_combination(filename, features):
    """

    """
    best_accuracy = 0
    best_combination = None

    # Genera todas las combinaciones posibles de características
    combinations = generate_combinations(features)

    # Itera sobre todas las combinaciones y entrena el modelo
    for combination in combinations:
        print(f"\nTesting combination: {combination}")
        train(filename, columns_to_drop=combination)


def main():
    if len(sys.argv) != 2:
        print('❗️ Usage: python3 optimize.py data*.csv')
        exit(1)

    file_path = sys.argv[1]
    print(f'\n🟡 File path:{file_path}\n')

    if not os.path.isfile(file_path):
        print('❌ Error: File not found')
        exit(1)

    features = ['Astronomy', 'History of Magic', 'Transfiguration', 'Charms', 'Flying',
                'Arithmancy', 'Care of Magical Creatures', 'Herbology', 'Potions',
                'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes']

    find_best_combination(file_path, features)


if __name__ == "__main__":
    main()
