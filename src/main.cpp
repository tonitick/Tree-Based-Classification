#include "data.h"
#include "tree.h"
#include <omp.h>
#include <time.h>
#include <stdio.h>

#include <assert.h>

int main() {
  // train data
  // int item_num = TRAINING_SET_SIZE;
  int item_num = 100000;
  vector<DataItem> train_data;
  string train_path = "../data/train_data.txt";
  double t1 = omp_get_wtime();
  getData(train_path, train_data, item_num, 0);
  double t2 = omp_get_wtime();
  printf("train data loaded. time cost = %lfs\n", t2 - t1);
  printf("train data size = %ld\n", train_data.size());
  // showData(train_data);

  //build trees
  GBDT gbdt(BOOSTING_SIZE, MAX_DEPTH, MAX_LEAVES);
  vector<int> items_index;
  vector<int> features_id;
  for(int i = 0; i < item_num; i++) {
    items_index.push_back(i);
  }
  for(int i = 201; i >= 1; i--) {
    features_id.push_back(i);
  }
  double t5 = omp_get_wtime();
  gbdt.build(train_data, items_index, features_id);
  double t6 = omp_get_wtime();
  printf("tree built. time cost = %lfs\n", t6 - t5);

  gbdt.show();

  //testing with train data
  printf("testing with train data\n");
  vector<vector<double> > result = gbdt.estimateTreeWise(train_data, 0);
  string result_path = "../data/result.txt";

  //output
  item_num = TESTING_SET_SIZE;
  vector<DataItem> test_data;
  string test_path = "../data/test_data.txt";
  double t7 = omp_get_wtime();
  getData(test_path, test_data, item_num, 1);
  double t8 = omp_get_wtime();
  printf("testing data loaded. time cost = %lfs\n", t8 - t7);
  printf("testing data size = %ld\n", test_data.size());
  vector<double> ouput = gbdt.estimate(test_data);
  string output_path = "../data/submission.txt";
  writeData(output_path, ouput);

  return 0;
}
