#include "data.h"
#include "tree.h"
#include <omp.h>
#include <time.h>
#include <stdio.h>

int main() {
  double t1 = omp_get_wtime();
  int item_num = 100000;

  string path = "../data/train_data.txt";
  vector<DataItem> data;
  getData(path, data, item_num);
  printf("data size = %ld\n", data.size());
  // showData(data);
  double t2 = omp_get_wtime();
  printf("data loaded. time cost = %lfs\n", t2 - t1);
  
  Forest forest(BAGGING_FREQUENCY, MAX_DEPTH, MAX_LEAVES);
  vector<int> indices;
  for(int i = 0; i < item_num; i++) {
    indices.push_back(i);
  }
  forest.build(data, indices);

  double t3 = omp_get_wtime();
  printf("tree built. time cost = %lfs\n", t3 - t2);

  forest.showForest();
  
  return 0;
}