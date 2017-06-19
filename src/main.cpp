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
  Forest forest(BOOSTING_SIZE, MAX_DEPTH, MAX_LEAVES);
  vector<int> indices;
  for(int i = 0; i < item_num; i++) {
    indices.push_back(i);
  }
  double t5 = omp_get_wtime();
  vector<vector<double> > result_ori = forest.build(train_data, indices);
  double t6 = omp_get_wtime();
  printf("tree built. time cost = %lfs\n", t6 - t5);

  forest.showForest();

  //testing1
  printf("testing result\n");
  vector<vector<double> > result1 = forest.estimateTreeWise(train_data, 0);
  string result1_path = "../data/result1.txt";
  writeDataTreeWise(result1_path, result1, 0);
  // printf("%d %d %d, %d\n", result_ori.size(), result1.size());
  assert(result_ori.size() == result1.size());
  assert(result_ori[0].size() == result1[0].size());
  printf("testing ori\n");
  string result_ori_path = "../data/result_ori.txt";
  writeDataTreeWise(result_ori_path, result_ori, 0);

  //testing
  item_num = TESTING_SET_SIZE;
  vector<DataItem> test_data;
  string test_path = "../data/test_data.txt";
  double t7 = omp_get_wtime();
  getData(test_path, test_data, item_num, 1);
  double t8 = omp_get_wtime();
  printf("testing data loaded. time cost = %lfs\n", t8 - t7);
  printf("testing data size = %ld\n", test_data.size());
  vector<vector<double> >result2 = forest.estimateTreeWise(test_data, 1);
  // double loss = 0;
  // for(int i = 0; i < result.size(); i++) {
  //   loss += (result[i] - train_data[i].label);
  // }
  // printf("averate loss = %lf\n", loss / item_num);
  string result2_path = "../data/result2.txt";
  writeDataTreeWise(result2_path, result2, 1);

  //output
  vector<double> ouput = forest.estimate(test_data);
  string output_path = "../data/submission.txt";
  writeData(output_path, ouput);

  // vector<DataItem> train_data;
  // string train_path = "../data/train_data.txt";
  // getData(train_path, train_data, -1, 0);
  // printf("data size = %d\n", train_data.size());
  // int count[FEATURE_NUMBER];
  // for(int i = 0; i < FEATURE_NUMBER; i++) {
  //   count[i] = 0;
  // }
  // for(int i = 0; i < train_data.size(); i++) {
  //   for(int j = 0; j < train_data[i].features.size(); j++) {
  //     count[train_data[i].features[j].feature_id]++;
  //   }
  // }
  // for(int i = 0; i < FEATURE_NUMBER; i++) {
  //   printf("feature %d: count = %d\n", i, count[i]);
  // }
  return 0;
}