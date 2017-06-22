#include "data.h"

#include <sstream>
#include <fstream>

#include <stdio.h>

void getData(string file_path, vector<DataItem>& data, int num, int train_test) {
  data.clear();

  string s;
  ifstream input_file(file_path.c_str());
  stringstream ss;
  int label;
  int feature_id;
  char split;
  double value;
  
  int count = 0;
  while(!input_file.eof()) {
    DataItem itemToAdd;

    getline(input_file,s);
    ss.clear();
    ss.str(s);

    ss >> label;
    itemToAdd.label = label;

    if(train_test == 0 || label == data.size()) { //remove duplicate item in testing set
      count++;
      while(1) {
        ss >> feature_id >> split >> value;
        if (ss.fail()) {
          break;
        }

        Feature featureToAdd;
        featureToAdd.feature_id = feature_id;
        featureToAdd.value = value;

        itemToAdd.features.push_back(featureToAdd);
      }

      data.push_back(itemToAdd);
    }

    if(num != -1 && count >= num) {
      break;
    }
  }

  input_file.close();
}

void writeData(string file_path, const vector<double>& data) {
  ofstream output_file(file_path.c_str());
  output_file << "id,label" << endl;
  for(int i = 0; i < data.size(); i++) {
    output_file << i << ',' << data[i] << endl;
  }

  output_file.close();
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


// below are functions for debugging, commented

// void showData(const vector<DataItem>& items) {
//   for(int i = 0; i < items.size(); i++) {
//     DataItem item = items[i];
//     printf("%d", item.label);
//     for(int j = 0; j < item.features.size(); j++) {
//       printf(" %d:%lf", item.features[j].feature_id, 
//           item.features[j].value);
//     }
//     printf("\n");
//   }
// }

// void writeDataTreeWise(string file_path, const vector<vector<double> >& data, int train_test) {
//   ofstream output_file(file_path.c_str());
//   output_file << "id";
//   if(data.size() && data[0].size()) {
//     int dataitem_size = data[0].size();
//     int tree_num;
//     if(train_test == 0) {
//       tree_num = (dataitem_size - 1) / 4;
//     }
//     else {
//       tree_num = dataitem_size / 3;
//     }

//     printf("tree number = %d\n", tree_num);

//     for(int i = 0; i < tree_num; i++) {
//       output_file << ",";
//       output_file << "tree" << i << "value,";
//       output_file << "tree" << i << "sum,";
//       output_file << "tree" << i << "estimate";
//       if(train_test == 0) {
//         output_file << ",tree" << i << "error";
//       }
//     }
//     if(train_test == 0) {
//       output_file << ",label" << endl;
//     }
//     else {
//       output_file << endl;
//     }

//     for(int i = 0; i < data.size(); i++) {
//       output_file << i;
//       for(int j = 0; j < dataitem_size; j++) {
//         output_file << ',' << data[i][j];
//       }
//       output_file << endl;
//     }
//   }
// }
