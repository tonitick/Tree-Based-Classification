#include "data.h"
#include "tree.h"
#include "bagging.h"
#include <omp.h>
#include <time.h>
#include <stdio.h>

#include <assert.h>

int main() {
  // train data
  vector<DataItem> train_data;
  string train_path = "../data/train_data.txt";
  double t1 = omp_get_wtime();
  getData(train_path, train_data, TRAINING_SET_SIZE, 0);
  double t2 = omp_get_wtime();
  printf("train data loaded. time cost = %lfs\n", t2 - t1);
  printf("train data size = %ld\n", train_data.size());

  //build
  Bagging bagging(TRAINING_SET_SIZE, FEATURE_NUMBER);
  for(int i = 0; i < BAGGING_SIZE; i++) {
    bagging.addOne(train_data ,TRAINING_SET_SIZE_FOR_TRAIN, FEATURE_NUM_FOR_TRAIN);
  }

  //test data
  vector<DataItem> test_data;
  string test_path = "../data/test_data.txt";
  double t7 = omp_get_wtime();
  getData(test_path, test_data, TESTING_SET_SIZE, 1);
  double t8 = omp_get_wtime();
  printf("testing data loaded. time cost = %lfs\n", t8 - t7);
  printf("testing data size = %ld\n", test_data.size());
  
  //output
  vector<double> ouput = bagging.estimate(test_data);
  string output_path = "../data/submission.txt";
  writeData(output_path, ouput);

  //cross validation
  bagging.crossValidation(train_data, 200000);

  return 0;
}
