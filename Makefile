# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Makefile                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dgerwig- <dgerwig-@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/06 14:07:52 by dgerwig-          #+#    #+#              #
#    Updated: 2024/06/01 18:27:03 by dgerwig-         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

DATA_TRAIN_FILE_PATH = './data/dataset_train.csv'
DATA_TEST_FILE_PATH  = './data/dataset_test.csv'
DATA_CLEAN_FILE_PATH = './data/dataset_train_clean.csv'

all:


plot: fclean describe histogram scatter pair

describe:
	@echo -e '\033[31mDATA ANALYSIS\033[0m'
	@echo -e '\033[31mDescribing data...\033[0m'
	@python3 src/utils/describe.py $(DATA_TRAIN_FILE_PATH)

histogram:
	@echo -e '\n\033[31mDATA VISUALIZATION\033[0m'
	@echo -e '\n\033[31mGenerating histogram...\033[0m'
	@python3 src/visual/histogram.py $(DATA_TRAIN_FILE_PATH)

scatter:
	@echo -e '\n\033[31mDATA VISUALIZATION\033[0m'
	@echo -e '\n\033[31mGenerating scatter plot...\033[0m'
	@python3 src/visual/scatter_plot.py $(DATA_TRAIN_FILE_PATH)

pair:
	@echo -e '\n\033[31mDATA VISUALIZATION\033[0m'
	@echo -e '\n\033[31mGenerating pair plot...\033[0m'
	@python3 src/visual/pair_plot.py $(DATA_TRAIN_FILE_PATH)


test: fclean train predict evaluate

train:
	@echo -e '\n\033[31mLOGISTIC REGRESSION\033[0m'
	@echo -e '\n\033[31mTraining...\033[0m'
	@python3 src/logreg/logreg_train.py $(DATA_TRAIN_FILE_PATH)

predict:
	@echo -e '\n\033[31mLOGISTIC REGRESSION\033[0m'
	@echo -e '\n\033[31mPredicting...\033[0m'
	@python3 src/logreg/logreg_predict.py $(DATA_TEST_FILE_PATH)

evaluate:
	@echo -e '\n\033[31mLOGISTIC REGRESSION\033[0m'
	@echo -e '\n\033[31mEvaluating...\033[0m'
	@python3 src/test/evaluate.py


optimize: fclean
	@echo -e '\n\033[31mOPTIMIZE SELECTED FEATURES\033[0m'
	@echo -e '\n\033[31mOptimizing...\033[0m'
	@python3 src/utils/optimize.py $(DATA_TRAIN_FILE_PATH)


express: fclean
	@python3 src/logreg/logreg_train.py $(DATA_TRAIN_FILE_PATH) --skip-input
	@python3 src/logreg/logreg_predict.py $(DATA_TEST_FILE_PATH) --skip-input
	@python3 src/test/evaluate.py --skip-input


req:
	@pip3 install -r requirements.txt


clean:

fclean: clean
	@echo -e '\n🟡 Cleaning up...\n'
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@rm -rf ./data/*_clean.csv
	@rm -rf ./data/params.csv
	@rm -rf ./data/houses.csv
	@rm -rf ./data/df_num.csv
	@if [ -d './plots' ]; then \
		mkdir -p ./plots_examples; \
		cp -r ./plots/* ./plots_examples/; \
		rm -rf ./plots; \
	fi

re:	fclean all

phony: all clean fclean re describe histogram scatter pair train predict evaluate optimize express requirements
