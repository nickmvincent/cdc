# About this repo
This is an archive of scripts used to run experiments in:

Vincent. N. and Hecht, B. 2021.
Can “Conscious Data Contribution” Help Users to Exert “Data Leverage” Against Technology Companies? To Appear in CSCW 2021 / PACM Computer-Supported Cooperative Work and Social Computing. 

This repo mainly has high-level scripts, as we used software from several other papers to implement various models. The relevant links to other software packages are below.

See the paper for more details on the experiments, but in short
"conscious data contribution" and "data strikes" experiments are very similar to the experiments run to produce performance vs. training dataset size "learning curves".

We simulate two companies, "Large Co." and "Small Co." and consider how ML performance changes as each company loses or gains data from a data strike or conscious data contribution campaign.


 One key difference is that consider performance both from an "objective" perspective (a fixed holdout set) and "subjective" perspective (we imagine each company must create a test set from the data available to it).


# ML-10M RecSys experiment code
We used libfm (http://libfm.org/) from Steffen Rendle.

Relevant code files: `shuf_ml-10m.py` (creates a single 90-10 train-test split, or 10 folds if modified), `run_libfm.py` (calls the libfm software), `eval_libfm.py` (evaluate model performance).

# Pinterest RecSys experiment code
https://github.com/nickmvincent/RecSys2019_DeepLearning_Evaluation

This a fork of code provided by:

Maurizio Ferrari Dacrema, Paolo Cremonesi, and Dietmar Jannach. 2019. Are we really making much progress? A worrying analysis of recent neural recommendation approaches. In Proceedings of the 13th ACM Conference on Recommender Systems (RecSys '19). Association for Computing Machinery, New York, NY, USA, 101–109. DOI:https://doi.org/10.1145/3298689.3347058


We added a new script, `run_single.py` to the forked repo. This script uses the Item-based KNN implementation from Dacrema et al. but runs data strike / CDC simulations.

One wrinkle unqiue to this task is that this approach does not generate recommendations for unseen users. For these users, we assume they receive performance equivalent to an unpersonalized baseline, "TopPop" (recommend most popular).

# CIFAR-10 experiment code
We ran these experiments in a Google Colab notebook (as these experiments used neural networks which benefit from GPUs)
Currently, we have a Google drive link to the .ipynb file below. In the future, we may try to set up a more interactive version, if there is interest.
https://colab.research.google.com/drive/1_RW8XHYBG3jDgTh43K3GAAO9Mn_8_4S8?usp=sharing

The original approach is from David Page, https://myrtle.ai/learn/how-to-train-your-resnet/, a top performer in the Dawnbench competition (https://dawn.cs.stanford.edu/benchmark/index.html).

# Wikipedia Toxic Comments experiment code
This code lives in this repo.

See `cdc_split.py` (implements the data prep from Kaggle user guocan https://www.kaggle.com/guocan/logistic-regression-with-words-and-char-n-g-13417e) and `run_cdc_sklearn.py` (driver code for training/testing).

In the paper, we note that 'The binary classification performance of the ML approach is very close to the average performance across the six categories'

Here are the results from the above kaggle notebook:
toxic is 0.978566839758313
severe_toxic is 0.9886024475623884
obscene is 0.990113763853626
threat is 0.9893489760024069
insult is 0.9826925321664285
identity_hate is 0.982642365633132
avg: 0.9853278208293825

We created a binary classification task by treating each comment with a positive label in any category as toxic.
Our binary class reuslts: 0.9745094019688552 for company-perspective (subjective), 0.9720217014497914 for fixed holdout (objective / hidden)

# Paper figures
See `compare_performance.py` (this is a .py file that runs as a Jupyter notebook via Vscode Jupyter notebook feature).
