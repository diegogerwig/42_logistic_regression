# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Makefile                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dgerwig- <dgerwig-@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/06 14:07:52 by dgerwig-          #+#    #+#              #
#    Updated: 2024/04/13 11:53:17 by dgerwig-         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

DATA_TRAIN_FILE_PATH = "./data/dataset_train.csv"
DATA_CLEAN_FILE_PATH = "./data/dataset_train_clean.csv"

all: describe histogram scatter pair train

describe:
	@echo "\n\033[31mDATA ANALYSIS\033[0m"
	@echo "\n\033[31mDescribing data...\033[0m"
	@python3 ./src/describe.py $(DATA_TRAIN_FILE_PATH)

histogram:
	@echo "\n\033[31mDATA VISUALIZATION\033[0m"
	@echo "\n\033[31mGenerating histogram...\033[0m"
	@if [ -f "$(DATA_CLEAN_FILE_PATH)" ]; \
	then \
		python3 ./src/histogram.py $(DATA_CLEAN_FILE_PATH); \
	else \
		python3 ./src/histogram.py $(DATA_TRAIN_FILE_PATH); \
	fi

scatter:
	@echo "\n\033[31mDATA VISUALIZATION\033[0m"
	@echo "\n\033[31mGenerating scatter plot...\033[0m"
	@if [ -f "$(DATA_CLEAN_FILE_PATH)" ]; \
	then \
		python3 ./src/scatter_plot.py $(DATA_CLEAN_FILE_PATH); \
	else \
		python3 ./src/scatter_plot.py $(DATA_TRAIN_FILE_PATH); \
	fi

pair:
	@echo "\n\033[31mDATA VISUALIZATION\033[0m"
	@echo "\n\033[31mGenerating pair plot...\033[0m"
	@if [ -f "$(DATA_CLEAN_FILE_PATH)" ]; \
	then \
		python3 ./src/pair_plot.py $(DATA_CLEAN_FILE_PATH); \
	else \
		python3 ./src/pair_plot.py $(DATA_TRAIN_FILE_PATH); \
	fi

train:
	@echo "\n\033[31mLOGISTIC REGRESSION\033[0m"
	@echo "\n\033[31mTraining...\033[0m"
	@if [ -f "$(DATA_CLEAN_FILE_PATH)" ]; \
	then \
		python3 ./src/logreg_train.py $(DATA_CLEAN_FILE_PATH); \
	else \
		python3 ./src/logreg_train.py $(DATA_TRAIN_FILE_PATH); \
	fi

clean:
	
fclean: clean
	@echo "\n🟡 Cleaning up...\n"
	@rm -rf **/__pycache__
	@rm -rf **/*_clean.csv
	@cp -r ./plots ./plots_example
	@rm -rf ./plots

re:	fclean all
	
phony: all clean fclean re describe histogram scatter pair train
