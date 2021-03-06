#include "tree.h"

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <vector>
#include <algorithm> //enable sort
#include <omp.h> //openmp parallelization

#include <fstream>
#include <iostream>
using namespace std;

struct ItemWithOneFeature {
  int index;
  Feature feature;
};

bool cmp(ItemWithOneFeature a, ItemWithOneFeature b) {
  return a.feature.value < b.feature.value;
}

bool cmp_index(ItemPack a, ItemPack b) {
  return a.item_index < b.item_index;
}

void GBDT::rearrange(vector<ItemPack>& itempacks, const NodePack& cur_nodepack, 
    int feature_id, int split_point, double left_value, double right_value,
    const vector<DataItem>& data) {
  //get rearrange order
  int item_number = cur_nodepack.end_index - cur_nodepack.start_index;
  vector<ItemWithOneFeature> index_feature(item_number);
  for(int i = 0; i < item_number; i++) {
    int pack_index = cur_nodepack.start_index + i;
    index_feature[i].index = pack_index;
    index_feature[i].feature = getFeature(data, itempacks[pack_index].item_index, feature_id);
  }
  sort(index_feature.begin(), index_feature.end(), cmp);
  
  //rearrange
  vector<ItemPack> temp;
  for(int i = 0; i < item_number; i++) {
    ItemPack itemPackToAdd = itempacks[index_feature[i].index];
    temp.push_back(itemPackToAdd);
  }
  for(int i = 0; i < item_number; i++) {
    int pack_index = cur_nodepack.start_index + i;
    itempacks[pack_index] = temp[i];
    if(pack_index <= split_point) {
      itempacks[pack_index].current_value = left_value;
    }
    else {
      itempacks[pack_index].current_value = right_value;
    }
  }
}

double getObj(double g, double h) {
  return -(1.0 / 2.0) * g * g / h;
}

double getVal(double g, double h) {
  return - g / h;
}

double sigmoid(double x) {
  return 1.0 / (1 + exp(-x));
}

Tree::Tree() {}

void Tree::setNode(int node_id, int feature_id, double partition_value) {
  nodes[node_id].feature_id = feature_id;
  nodes[node_id].partition_value = partition_value;
}

TreeNode Tree::getNode(int node_id) {
  return nodes[node_id];
}

int Tree::getLeftNodeId(int node_id) {
  return left[node_id];
}
int Tree::getRightNodeId(int node_id) {
  return right[node_id];
}

int Tree::addNode(int parent_id, int left_right, double nv) {
  TreeNode nodeToAdd;
  nodeToAdd.feature_id = -1;
  nodeToAdd.node_value = nv;
  if(parent_id >= 0) { //if not root node
    nodeToAdd.level = nodes[parent_id].level + 1;
  }
  else { //root node
    nodeToAdd.level = 0;
  }
  int node_id = nodes.size();
  nodes.push_back(nodeToAdd);
  left.push_back(-1);
  right.push_back(-1);

  if(parent_id >= 0) { //if not root node
    if(left_right == 0) {
      left[parent_id] = node_id;
    }
    else {
      right[parent_id] = node_id;
    }
  }

  return node_id;
}

GBDT::GBDT(int tn, int td, int ln) {
  tree_num = tn;
  tree_depth = td;
  leave_num = ln;
}

int GBDT::splitNode(NodePack& cur_nodepack, vector<ItemPack>& itempacks,
    const vector<DataItem>& data, const vector<int> features_id, int leaves) {
  if(leaves + 1 >= leave_num) {
    return -1;
  }
  int item_number = cur_nodepack.end_index - cur_nodepack.start_index;
  if(item_number <= 1 || cur_nodepack.level >= MAX_DEPTH) { //interval with no item
    return -1;
  }

  //calculate original value
  double G = 0.0, H = 0.0;
  for(int i = 0; i < item_number; i++) {
    int itempack_index = cur_nodepack.start_index + i;
    G += itempacks[itempack_index].first_order;
    H += itempacks[itempack_index].second_order;
  }
  double ori_obj = getObj(G, H);
  double ori_val = getVal(G, H);
  for(int i = 0; i < item_number; i++) { //set current value
    int itempack_index = cur_nodepack.start_index + i;
    itempacks[itempack_index].current_value = ori_val;
  }

  //feature info
  int feature_size = features_id.size();
  vector<double> gains(feature_size);
  vector<int> split_points(feature_size);
  vector<double> left_values(feature_size);
  vector<double> right_values(feature_size);
  vector<double> partition_values(feature_size);
  //find split feature with parallelization
  #pragma omp parallel for
  for(int feature_index = 0; feature_index < feature_size; feature_index++) {
    int feature_id = features_id[feature_index];
    
    //get features of each item
    vector<ItemWithOneFeature> index_feature(item_number);
    for(int i = 0; i < item_number; i++) {
      int itempack_index = cur_nodepack.start_index + i;
      index_feature[i].index = itempack_index;
      index_feature[i].feature = getFeature(data, itempacks[itempack_index].item_index, feature_id);
    }
    //sort items by feature value
    sort(index_feature.begin(), index_feature.end(), cmp);

    //initialization
    double gl = 0.0, hl = 0.0, gr = G, hr = H;
    double opt_obj = ori_obj;
    double left_value, right_value;
    int split_point = -1;
    double partition_value;
    //find optimal split point
    int index = 0;
    while(index < item_number - 1) {
      //find the range that share the same feature value
      int start = index, end = index + 1;
      while(end < item_number - 1 
          && index_feature[end].feature.value == index_feature[start].feature.value) {
        end++;
      }
      int range_num = end - start;

      //find split point
      for(int j = start; j < end; j++) {
        int itempack_index = index_feature[j].index;
        gl += itempacks[itempack_index].first_order;
        hl += itempacks[itempack_index].second_order;
        gr -= itempacks[itempack_index].first_order;
        hr -= itempacks[itempack_index].second_order;

        double fraction = (double)(j - start + 1) / (double)range_num;
        if(fraction > 0.1 && fraction < 0.9) {
          continue;
        }

        if(getObj(gl, hl) + getObj(gr, hr) < opt_obj) {
          if(fraction <= 0.1) {
            if(start != 0) {//guarantee not to split point that is out of range
              split_point = cur_nodepack.start_index + start - 1;
              partition_value = index_feature[start - 1].feature.value;
              opt_obj = getObj(gl, hl) + getObj(gr, hr);
              left_value = getVal(gl, hl);
              right_value = getVal(gr, hr);
            }
          }
          else if(fraction >= 0.9) {
            if(index_feature[end - 1].feature.value
                != index_feature[end].feature.value) {//guarantee not to split at end - 1 when its value is equal to that of end
              split_point = cur_nodepack.start_index + end - 1;
              partition_value = index_feature[end - 1].feature.value;
              opt_obj = getObj(gl, hl) + getObj(gr, hr);
              left_value = getVal(gl, hl);
              right_value = getVal(gr, hr);
            }
          }
          else {
            assert(0);
          }

        }
      }
      
      index = end;
    }

    //record feature info
    gains[feature_index] = ori_obj - opt_obj;
    split_points[feature_index] = split_point;
    left_values[feature_index] = left_value;
    right_values[feature_index] = right_value;
    partition_values[feature_index] = partition_value;
  }

  //split
  int split_feature_index = -1;
  bool flag = 0;
  double gain;
  for(int feature_index = 0; feature_index < feature_size; feature_index++) {
    if(split_points[feature_index] != -1) {
      if(flag == 0) {
        flag = 1;
        split_feature_index = feature_index;
        gain = gains[feature_index];
      }
      else if(gain < gains[feature_index]) {
        split_feature_index = feature_index;
        gain = gains[feature_index];
      }
    }
  }
  if(split_feature_index == -1) {//no suitable feature to split node
    return -1;
  }
  else {//split node
    int split_feature = features_id[split_feature_index];
    int split_point = split_points[split_feature_index];
    double gain = gains[split_feature_index];
    double left_value = left_values[split_feature_index];
    double right_value = right_values[split_feature_index];
    double partition_value = partition_values[split_feature_index];

    printf("split point = %d, gain = %lf, split feature = %d, partition value = %lf, lv  = %lf, rv = %lf, ln = %d, rn = %d\n",
        split_point, gain, split_feature, partition_value, left_value, right_value, split_point - cur_nodepack.start_index + 1,
        cur_nodepack.end_index - split_point - 1);
    
    rearrange(itempacks, cur_nodepack, split_feature,
        split_point, left_value, right_value, data);
    
    cur_nodepack.feature_id = split_feature;
    // assert(itempacks[split_point].current_value == left_value);
    // assert(itempacks[cur_nodepack.start_index].current_value == left_value);
    // assert(itempacks[cur_nodepack.end_index - 1].current_value == right_value);
    
    return split_point;
  }
}

double GBDT::build(const vector<DataItem>& data, const vector<int> items_index,
    const vector<int> features_id) {
  trees.clear();

  //initialization: all items are assigned 0
  double loss = 0.0;
  
  vector<ItemPack> itempacks;
  for(int i = 0; i < items_index.size(); i++) {
    ItemPack itemPackToAdd;
    itemPackToAdd.item_index = items_index[i];
    itemPackToAdd.current_sum = 0.0;
    // itemPackToAdd.first_order = 2.0 * (- data[items_index[i]].label);
    // itemPackToAdd.second_order = 2.0;
    itemPackToAdd.first_order = sigmoid(0.0) - data[items_index[i]].label;
    itemPackToAdd.second_order = exp(-0.0) * sigmoid(0.0) * sigmoid(0.0);
    itempacks.push_back(itemPackToAdd);
    
    loss += (sigmoid(0.0) - data[items_index[i]].label) * (sigmoid(0.0) - data[items_index[i]].label);
  }
  printf("data size = %ld\n", itempacks.size());
  printf("original average loss = %lf\n", loss / itempacks.size());

  //build trees one by one
  double rate = 1.0;
  for(int tree_id = 0; tree_id < tree_num; tree_id++) {
    int leaves = 1;
    Tree treeToAdd;
    //add root to current tree
    int root_id = treeToAdd.addNode(-1, -1, 0.0); 
    vector<NodePack> nodepacks;
    NodePack rootNodePack;
    rootNodePack.node_id = root_id;
    rootNodePack.level = 0;
    rootNodePack.start_index = 0;
    rootNodePack.end_index = items_index.size();
    rootNodePack.feature_id = -1; //not splited yet
    nodepacks.push_back(rootNodePack);

    //build current tree from root node
    int nodepack_index = 0;
    while(nodepack_index < nodepacks.size()) {
      NodePack cur_nodepack = nodepacks[nodepack_index];
      nodepack_index++;
      
      //split
      int split_point = splitNode(cur_nodepack, itempacks, data, features_id, leaves);
      if(split_point != -1) {
        leaves++;
        
        //update splited node
        double partition_value = getFeature(data, itempacks[split_point].item_index, cur_nodepack.feature_id).value;
        treeToAdd.setNode(cur_nodepack.node_id, cur_nodepack.feature_id, partition_value);
        
        //add two new nodes
        double left_value = itempacks[cur_nodepack.start_index].current_value;
        double right_value = itempacks[cur_nodepack.end_index - 1].current_value;
        int left_id = treeToAdd.addNode(cur_nodepack.node_id, 0, left_value);
        int right_id = treeToAdd.addNode(cur_nodepack.node_id, 1, right_value);
        NodePack left_Node, right_Node;
        left_Node.node_id = left_id;
        left_Node.level = cur_nodepack.level + 1;
        left_Node.start_index = cur_nodepack.start_index;
        left_Node.end_index = split_point + 1;
        left_Node.feature_id = -1;
        right_Node.node_id = right_id;
        right_Node.level = cur_nodepack.level + 1;
        right_Node.start_index = split_point + 1;
        right_Node.end_index = cur_nodepack.end_index;
        right_Node.feature_id = -1;
        nodepacks.push_back(left_Node);
        nodepacks.push_back(right_Node);
      }
    }

    trees.push_back(treeToAdd);

    //update sum, g & h
    sort(itempacks.begin(), itempacks.end(), cmp_index);
    loss = 0.0;
    for(int i = 0; i < itempacks.size(); i++) {
      itempacks[i].current_sum += rate * itempacks[i].current_value;
      // itempacks[i].first_order = 2.0 * (itempacks[i].current_sum - data[itempacks[i].item_index].label);
      // itempacks[i].second_order = 2.0;
      double y = itempacks[i].current_sum;
      
      itempacks[i].first_order = sigmoid(y) - data[itempacks[i].item_index].label;
      itempacks[i].second_order = exp(-y) * sigmoid(y) * sigmoid(y);
      
      loss += (sigmoid(y) - data[itempacks[i].item_index].label) * (sigmoid(y) - data[itempacks[i].item_index].label);
    }

    //rate decaying
    rate *= DECAYING_RATE;

    // printf("data size = %ld\n", itempacks.size());
    printf("average loss of previous %d trees: %lf\n", tree_id + 1, loss / itempacks.size());
  }

  return loss / itempacks.size();
}

vector<double> GBDT::estimate(const vector<DataItem>& data) {
  vector<double> result(data.size());

  //estimate each data item with parallelization
  #pragma omp parallel for
  for(int i = 0; i < data.size(); i++) {
    double value = 0.0;
    double rate = 1.0;
    for(int tree_id = 0; tree_id < trees.size(); tree_id++) {
      int cur_node = 0; //root node initially
      int tar_node = 0; //root node initially

      //find the node that the data item is assigned to
      while(cur_node != -1) {
        int split_feature = trees[tree_id].getNode(cur_node).feature_id;
        double partition_value = trees[tree_id].getNode(cur_node).partition_value;
        double feature_value = getFeature(data, i, split_feature).value;
        if(feature_value <= partition_value) {
          tar_node = cur_node;
          cur_node = trees[tree_id].getLeftNodeId(cur_node);
        }
        else {
          tar_node = cur_node;
          cur_node = trees[tree_id].getRightNodeId(cur_node);
        }
      }

      //add value
      value += rate * trees[tree_id].getNode(tar_node).node_value;

      //rate decaying
      rate *= DECAYING_RATE;
    }
    
    result[i] = sigmoid(value);
  }
  
  return result;
}

void Tree::logModel(string file_path) {
  ofstream log_file;
  log_file.open(file_path.c_str(), ios::app);

  for(int i = 0; i < nodes.size(); i++) {
    log_file << "node = " << i << ":" << endl
             << "left node = " << left[i] << ", " 
             << "right node = " << right[i] << ", "
             << "feature id = " << nodes[i].feature_id << ", "
             << "partition value = " << nodes[i].partition_value << ", "
             << "node value = " << nodes[i].node_value << ", "
             << "level = " << nodes[i].level << endl;
  }

  log_file.close();
}

void GBDT::logModel(string file_path) {
  ofstream log_file;

  for(int i = 0; i < trees.size(); i++) {
    log_file.open(file_path.c_str(), ios::app);
    log_file << "tree " << i << ":" << endl;
    log_file.close();
    trees[i].logModel(file_path);
  }
}

// below are functions for debugging, commented

// vector<vector<double> > GBDT::build(const vector<DataItem>& data, vector<int> items_index) {
//   vector<vector<double> > result;
//   for(int i = 0; i < items_index.size(); i++) {
//     vector<double> tmp(tree_num * 4 + 1);
//     result.push_back(tmp);
//   }

//   trees.clear();
//   //initialization: all items are assigned 0
//   double loss = 0.0;
//   vector<ItemPack> itempacks;
//   for(int i = 0; i < items_index.size(); i++) {
//     ItemPack itemPackToAdd;
//     itemPackToAdd.item_index = items_index[i];
//     itemPackToAdd.current_sum = 0.0;
//     // itemPackToAdd.first_order = 2.0 * (- data[items_index[i]].label);
//     // itemPackToAdd.second_order = 2.0;
//     itemPackToAdd.first_order = sigmoid(0.0) - data[items_index[i]].label;
//     itemPackToAdd.second_order = exp(-0.0) * sigmoid(0.0) * sigmoid(0.0);
//     // printf("first = %lf, second = %lf\n", itemPackToAdd.first_order, itemPackToAdd.second_order);
//     itempacks.push_back(itemPackToAdd);
    
//     loss += (sigmoid(0.0) - data[items_index[i]].label) * (sigmoid(0.0) - data[items_index[i]].label);
//   }
//   printf("data size = %d\n", itempacks.size());
//   printf("original average loss = %lf\n", loss / itempacks.size());
//   loss = 0.0;

//   //build trees one by one
//   for(int tree_id = 0; tree_id < tree_num; tree_id++) {
//     int leaves = 1;
//     Tree treeToAdd;
//     //add root to current tree
//     int root_id = treeToAdd.addNode(-1, -1, 0.0); 
//     vector<NodePack> nodepacks;
//     NodePack rootNodePack;
//     rootNodePack.node_id = root_id;
//     rootNodePack.level = 0;
//     rootNodePack.start_index = 0;
//     rootNodePack.end_index = items_index.size();
//     rootNodePack.feature_id = -1; //not splited yet
//     nodepacks.push_back(rootNodePack);

//     //build current tree from root node
//     int nodepack_index = 0;
//     while(nodepack_index < nodepacks.size()) {
//       NodePack cur_nodepack = nodepacks[nodepack_index];
//       nodepack_index++;
      
//       int split_point;
//       split_point = splitNode(cur_nodepack, itempacks, data, leaves);
//       if(split_point != -1) {
//         leaves++;
//         //update splited node
//         double partition_value = getFeature(data, itempacks[split_point].item_index, cur_nodepack.feature_id).value;
//         treeToAdd.setNode(cur_nodepack.node_id, cur_nodepack.feature_id, partition_value);
        
//         //add two new nodes
//         double left_value = itempacks[cur_nodepack.start_index].current_value;
//         double right_value = itempacks[cur_nodepack.end_index - 1].current_value;
//         int left_id = treeToAdd.addNode(cur_nodepack.node_id, 0, left_value);
//         int right_id = treeToAdd.addNode(cur_nodepack.node_id, 1, right_value);
//         NodePack left_Node, right_Node;
//         left_Node.node_id = left_id;
//         left_Node.level = cur_nodepack.level + 1;
//         left_Node.start_index = cur_nodepack.start_index;
//         left_Node.end_index = split_point + 1;
//         left_Node.feature_id = -1;
//         right_Node.node_id = right_id;
//         right_Node.level = cur_nodepack.level + 1;
//         right_Node.start_index = split_point + 1;
//         right_Node.end_index = cur_nodepack.end_index;
//         right_Node.feature_id = -1;
//         nodepacks.push_back(left_Node);
//         nodepacks.push_back(right_Node);
//       } 
//     }

//     trees.push_back(treeToAdd);

//     //update sum, g & h
//     sort(itempacks.begin(), itempacks.end(), cmp_index);
//     double loss = 0.0;
//     for(int i = 0; i < itempacks.size(); i++) {
//       // printf("itempack size = %d\n", itempacks.size());
//       itempacks[i].current_sum += itempacks[i].current_value;
//       // printf("%d\n", itempacks[i].item_index);
//       // itempacks[i].first_order = 2.0 * (itempacks[i].current_sum - data[itempacks[i].item_index].label);
//       // itempacks[i].second_order = 2.0;
//       double y = itempacks[i].current_sum;
//       assert(itempacks[i].item_index == i);
//       itempacks[i].first_order = sigmoid(y) - data[itempacks[i].item_index].label;
//       itempacks[i].second_order = exp(-y) * sigmoid(y) * sigmoid(y);
      
//       loss += (sigmoid(y) - data[itempacks[i].item_index].label) * (sigmoid(y) - data[itempacks[i].item_index].label);
      
//       result[i][tree_id * 4] = itempacks[i].current_value;
//       result[i][tree_id * 4 + 1] = y;
//       result[i][tree_id * 4 + 2] = sigmoid(y);
//       result[i][tree_id * 4 + 3] = (sigmoid(y) - data[itempacks[i].item_index].label) * (sigmoid(y) - data[itempacks[i].item_index].label);
//       if(tree_id == 0) {
//         result[i][tree_num * 4] = data[i].label;
//       }
//     }
//     printf("data size = %d\n", itempacks.size());
//     printf("average loss = %lf\n", loss / itempacks.size());
//   }

//   return result;
// }

// vector<vector<double> > GBDT::estimateTreeWise(const vector<DataItem>& data, int train_test) {
//   vector<vector<double> > result;
  
//   vector<double> for_print(trees.size());
//   for(int i = 0; i< for_print.size(); i++) {
//     for_print[i] = 0.0;
//   }

//   for(int i = 0; i < data.size(); i++) {
//     vector<double> result_item;
//     double sum = 0.0;
//     for(int tree_id = 0; tree_id < trees.size(); tree_id++) {

//       int cur_node = 0; //root node initially
//       int tar_node = 0; //root node initially

//       //find the node that the data item is assigned to
//       while(cur_node != -1) {
//         int split_feature = trees[tree_id].getNode(cur_node).feature_id;
//         double partition_value = trees[tree_id].getNode(cur_node).partition_value;
//         double feature_value = getFeature(data, i, split_feature).value;

//         if(feature_value <= partition_value) {
//           tar_node = cur_node;
//           cur_node = trees[tree_id].getLeftNodeId(cur_node);
//         }
//         else {
//           tar_node = cur_node;
//           cur_node = trees[tree_id].getRightNodeId(cur_node);
//         }
//       }

//       //add value
//       double value = trees[tree_id].getNode(tar_node).node_value; //value of currnet tree
//       sum += value; //sum
//       result_item.push_back(value);
//       result_item.push_back(sum);
//       result_item.push_back(sigmoid(sum)); //estimate value
//       if(train_test == 0) {
//         result_item.push_back((sigmoid(sum) - data[i].label) * (sigmoid(sum) - data[i].label));
//         for_print[tree_id] += (sigmoid(sum) - data[i].label) * (sigmoid(sum) - data[i].label);
//       }
//     }
//     if(train_test == 0) {
//       result_item.push_back(data[i].label);
//     }

//     result.push_back(result_item);
//   }

//   if(train_test == 0) {
//     printf("data size = %d\n", data.size());
//     for(int i = 0; i < for_print.size(); i++) {
//       printf("average loss of tree %d: %f\n", i, for_print[i] / data.size());
//     }
//   }
//   return result;
// }
