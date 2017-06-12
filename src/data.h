#ifndef __DATA_H__
#define __DATA_H__

#include "def.h"
#include <vector>
#include <string>
using namespace std;

struct Feature {
  int feature_id;
  double value;  
};

struct DataItem {
  int label;
  vector<Feature> features;
};

void getData(string file_path, vector<DataItem>& items, int num);
Feature getFeature(const vector<DataItem>& data, int item_id, int feature_id);

void showData(const vector<DataItem>& items);

#endif
