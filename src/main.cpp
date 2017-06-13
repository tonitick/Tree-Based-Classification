#include "data.h"
#include "tree.h"
#include <omp.h>
#include <time.h>
#include <stdio.h>

int main() {
  //train data
  // int item_num = 1866819;
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
  Forest forest(BAGGING_FREQUENCY, MAX_DEPTH, MAX_LEAVES);
  vector<int> indices;
  for(int i = 0; i < item_num; i++) {
    indices.push_back(i);
  }
  double t5 = omp_get_wtime();
  forest.build(train_data, indices);
  double t6 = omp_get_wtime();
  printf("tree built. time cost = %lfs\n", t6 - t5);

  forest.showForest();

  //testing
  item_num = 282796;
  vector<DataItem> test_data;
  string test_path = "../data/test_data.txt";
  double t7 = omp_get_wtime();
  getData(test_path, test_data, item_num, 0);
  double t8 = omp_get_wtime();
  printf("testing data loaded. time cost = %lfs\n", t8 - t7);
  printf("testing data size = %ld\n", test_data.size());
  vector<double> result = forest.estimate(test_data);
  // double loss = 0;
  // for(int i = 0; i < result.size(); i++) {
  //   loss += (result[i] - train_data[i].label);
  // }
  // printf("averate loss = %lf\n", loss / item_num);
  string output_path = "../data/submission.txt";
  writeData(output_path, result);

  return 0;
}