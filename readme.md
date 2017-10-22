# Tree Based Classification
A C++ implement of GBDT + Bagging Classification.  

# Data
Data for this program is available on [Large-scale classification-SYSU-2017](https://inclass.kaggle.com/c/large-scale-classification-sysu-2017), where we can also find the description of this classification task.

# Model Parameter
Modify model parameters in `src/def.h`  

  * BAGGING_SIZE: bagging size
  * BOOSTING_SIZE: boosting size
  * MAX_LEAVES: max leaves of one tree
  * MAX_DEPTH: max depth of one tree
  * DECAYING_RATE: decaying rate along the out put of boosting trees
  * FEATURE_NUMBER: feature number of data
  * FEATURE_NUM_FOR_TRAIN: number of randomly selected features for each bagging
  * TRAINING_SET_SIZE: number of items in training set
  * TRAINING_SET_SIZE_FOR_TRAIN: number of randomly selected training items for each bagging
  * TESTING_SET_SIZE: number of items in testing set
  * CROSS_VALIDATION_SIZE: number of training set items used to perform cross validation
  * SCATTER_RATIO: ratio of scatter number to size of data, used to generate random subset of training items and features

# Openmp Parallelization
This program use Openmp to enable parallelization in:  

  * finding feature to split nodes
  * generating result of testing set

To disable parallelization, modify `USE_OPENMP := 1` to `USE_OPENMP := 0` in Makfile, and remake.

# Run the Program

  * `cd src`
  * `make`
  * `./main.out`
