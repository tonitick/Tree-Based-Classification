#include "data.h"

#include <sstream>
#include <fstream>

// #include <stdio.h>

void getData(string file_path, vector<DataItem>& data) {
  data.clear();

  string s;
  ifstream input_file(file_path.c_str());
  stringstream ss;
  int label;
  int feature_id;
  char split;
  double value;
  
  while(!input_file.eof()) {
    DataItem itemToAdd;

    getline(input_file,s);
    ss.clear();
    ss.str(s);

    ss >> label;
    itemToAdd.label = label;

    while(1) {
      ss >> feature_id >> split >> value;
      if (ss.fail()) 
        break;

      Feature featureToAdd;
      featureToAdd.feature_id = feature_id;
      featureToAdd.value = value;

      itemToAdd.features.push_back(featureToAdd);
    }

    data.push_back(itemToAdd);
  }
}

Feature getFeature(const vector<DataItem>& data, int item_id, int feature_id) {
  DataItem item = data[item_id];
  for(int i = 0; i < item.features.size(); i++) {
    Feature feature = item.features[i];
    if(feature.feature_id == feature_id) {
      return feature;
    }
  }

  // not found, feature value = 0
  Feature feature;
  feature.feature_id = feature_id;
  feature.value = 0.0;
  return feature;
}

void showData(const vector<DataItem>& items) {
  for(int i = 0; i < items.size(); i++) {
    DataItem item = items[i];
    printf("%d", item.label);
    for(int j = 0; j < item.features.size(); j++) {
      printf(" %d:%lf", item.features[j].feature_id, 
          item.features[j].value);
    }
    printf("\n");
  }
}
