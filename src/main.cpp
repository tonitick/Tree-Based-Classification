#include "data.h"
#include "tree.h"
#include "bagging.h"
#include <time.h>
#include <stdio.h>
#include <omp.h>

int main() {
  // train data
  vector<DataItem> train_data;
  string train_path = "../data/train_data.txt";
  printf("loading training data...\n");
#ifdef _OPENMP
  double t1 = omp_get_wtime();
#else
  clock_t t1 = clock();
#endif
  getData(train_path, train_data, TRAINING_SET_SIZE, 0);
#ifdef _OPENMP
  double t2 = omp_get_wtime();
#else
  clock_t t2 = clock();
#endif
  double time1 = t2 - t1;
#ifndef _OPENMP
  time1 /= 1000000.0;
#endif
  printf("training data loaded. time cost = %lfs\n", time1);
  printf("training data size = %ld\n", train_data.size());

  //build
  printf("training GBDT models...\n");
#ifdef _OPENMP
  double t3 = omp_get_wtime();
#else
  clock_t t3 = clock();
#endif
  Bagging bagging(TRAINING_SET_SIZE, FEATURE_NUMBER);
  for(int i = 0; i < BAGGING_SIZE; i++) {
    printf("bagging %d:\n", i);
    bagging.addOne(train_data ,TRAINING_SET_SIZE_FOR_TRAIN, FEATURE_NUM_FOR_TRAIN);
  }
#ifdef _OPENMP
  double t4 = omp_get_wtime();
#else
  clock_t t4 = clock();
#endif
  double time2 = t4 - t3;
#ifndef _OPENMP
  time2 /= 1000000.0;
#endif
  printf("training completed. time cost = %lfs\n", time2);

  //test data
  vector<DataItem> test_data;
  string test_path = "../data/test_data.txt";
  printf("loading testing data...\n");
#ifdef _OPENMP
  double t5 = omp_get_wtime();
#else
  clock_t t5 = clock();
#endif
  getData(test_path, test_data, TESTING_SET_SIZE, 1);
#ifdef _OPENMP
  double t6 = omp_get_wtime();
#else
  clock_t t6 = clock();
#endif
  double time3 = t6 - t5;
#ifndef _OPENMP
  time3 /= 1000000.0;
#endif
  printf("testing data loaded. time cost = %lfs\n", time3);
  printf("testing data size = %ld\n", test_data.size());
  
  //cross validation
  bagging.crossValidation(train_data, CROSS_VALIDATION_SIZE);
  
  //output
  vector<double> ouput = bagging.estimate(test_data);
  string output_path = "../data/submission.txt";
  writeData(output_path, ouput);

  return 0;
}
