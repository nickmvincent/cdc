# About this repo
This is an archive of scripts used to run experiments in Nicholas Vincent and Brent Hecht's CSCW paper on "Conscious Data Contribution".

Vincent. N. and Hecht, B. 2021.
Can “Conscious Data Contribution” Help Users to Exert “Data Leverage” Against Technology Companies? To Appear in CSCW 2021 / PACM Computer-Supported Cooperative Work and Social Computing. 

This repo mainly has high-level scripts, as we used software from several other papers to implement various models. Relevant links are below


# ML-10M RecSys experiment code
We used libfm (http://libfm.org/).

Relevant code files: `shuf_ml-10m.py` (creates a single 90-10 train-test split, or 10 folds if modified), `run_libfm.py` (calls the libfm software), `eval_libfm.py` (evaluate model performance).

# Pinterest RecSys experiment code
https://github.com/nickmvincent/RecSys2019_DeepLearning_Evaluation
* TODO: describe which script in `scripts/` was used to 

# CIFAR-10 experiment code
We ran these experiments in a Google Colab notebook (as these experiments used neural networks which benefit from GPUs)
Currently, we have a Google drive link to the .ipynb file below. In the future, we may try to set up a more interactive version, if there is interest.
https://colab.research.google.com/drive/1_RW8XHYBG3jDgTh43K3GAAO9Mn_8_4S8?usp=sharing

# Wikipedia Toxic Comments experiment code
This code lives in this repo.

See `cdc_split.py` (implements the data prep from Kaggle user guocan https://www.kaggle.com/guocan/logistic-regression-with-words-and-char-n-g-13417e) and `run_cdc_sklearn.py` (driver code for training/testing).

# Paper figures
See `compare_performance.py` (this is a .py file that runs as a Jupyter notebook via Vscode Jupyter notebook feature).
