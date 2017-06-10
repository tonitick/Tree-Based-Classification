#include "data.h"
#include <omp.h>
#include <time.h>
#include <stdio.h>

int main() {
  double t1 = omp_get_wtime();

  string path = "../data/train_data.txt";
  vector<DataItem> items;
  getData(path, items);

  double t2 = omp_get_wtime();

  printf("%lfms\n", (t2 - t1) * 1000.0);

  return 0;
}