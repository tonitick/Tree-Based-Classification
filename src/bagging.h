#ifndef __BAGGING_H__
#define __BAGGING_H__

#include "tree.h"

class Bagging {
 private:
  int bagging_size;
  vector<double> losses;
  vector<GBDT> gbdts;
  vector<int> item_candidate;
  vector<int> feature_cadidate;

  void scatter(vector<int>& input); //used to generate random indices
  void swap(vector<int>& input, int i, int j); //used by scatter

 public:
  Bagging(int data_size, int feature_size);
  void addOne(const vector<DataItem>& data, int item_num, int feature_num);
  vector<double> estimate(const vector<DataItem>& data);

  void crossValidation(const vector<DataItem>& data, int val_size);
  void logModel(string file_path);
};

#endif