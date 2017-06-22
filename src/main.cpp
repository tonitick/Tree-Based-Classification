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
  getData(train_path, train_data, TRAINING_SET_SIZE, 0);
  printf("training data loaded.\n");
  printf("training data size = %ld\n", train_data.size());

  
  //build
  printf("training GBDT models...\n");
#ifdef _OPENMP
  double t1 = omp_get_wtime();
#else
  clock_t t1 = clock();
#endif
  Bagging bagging(TRAINING_SET_SIZE, FEATURE_NUMBER);
  for(int i = 0; i < BAGGING_SIZE; i++) {
    printf("bagging %d:\n", i);
    bagging.addOne(train_data ,TRAINING_SET_SIZE_FOR_TRAIN, FEATURE_NUM_FOR_TRAIN);
  }
#ifdef _OPENMP
  double t2 = omp_get_wtime();
#else
  clock_t t2 = clock();
#endif
  double time1 = t2 - t1;
#ifndef _OPENMP
  time1 /= 1000000.0;
#endif
  printf("training completed. time cost = %lfs\n", time1);

  
  //cross validation
  printf("performing cross validation...\n");
#ifdef _OPENMP
  double t3 = omp_get_wtime();
#else
  clock_t t3 = clock();
#endif
  bagging.crossValidation(train_data, CROSS_VALIDATION_SIZE);
#ifdef _OPENMP
  double t4 = omp_get_wtime();
#else
  clock_t t4 = clock();
#endif
  double time2 = t4 - t3;
#ifndef _OPENMP
  time2 /= 1000000.0;
#endif
  printf("cross validation completed. time cost = %lfs\n", time2);


  //test data
  vector<DataItem> test_data;
  string test_path = "../data/test_data.txt";
  printf("loading testing data...\n");
  getData(test_path, test_data, TESTING_SET_SIZE, 1);
  printf("testing data loaded");
  printf("testing data size = %ld\n", test_data.size());
  
  
  //generate result
  printf("generating result...\n");
#ifdef _OPENMP
  double t5 = omp_get_wtime();
#else
  clock_t t5 = clock();
#endif
  vector<double> ouput = bagging.estimate(test_data);
  string output_path = "../data/submission.txt";
#ifdef _OPENMP
  double t6 = omp_get_wtime();
#else
  clock_t t6 = clock();
#endif
  double time3 = t6 - t5;
#ifndef _OPENMP
  time3 /= 1000000.0;
#endif
  printf("result generated. time cost = %lfs\n", time3);
  
  
  //write result
  printf("writing result...\n");
  writeData(output_path, ouput);
  printf("result written.\n");

  //log model
  printf("logging model...\n");
  string log_path("../model/log.txt");
  bagging.logModel(log_path);
  printf("model logged.\n");

  return 0;
}
