#include "tree.h"

#include <stdio.h>
#include <assert.h>
#include <vector>
#include <algorithm>
using namespace std;

struct ItemWithOneFeature {
  int index;
  Feature feature;
};

bool cmp(ItemWithOneFeature a, ItemWithOneFeature b) {
  return a.feature.value < b.feature.value;
}

void rearrange(vector<ItemPack>& itempacks, NodePack& cur_nodepack, 
    int feature_id, int split_point, int left_value, int right_value,
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
  for(int i = 0; i < item_number; i++) {
    int pack_index = cur_nodepack.start_index + i;
    if(pack_index != index_feature[i].index) {
      int target_index = index_feature[i].index;
      ItemPack temp = itempacks[pack_index];
      itempacks[pack_index] = itempacks[target_index];
      itempacks[target_index] = temp;
      if(pack_index <= split_point) {
        itempacks[pack_index].current_value = left_value;
      }
      else {
        itempacks[pack_index].current_value = right_value;
      }
    }
  }
}

double getObj(double g, double h) {
  return -(1.0 / 2.0) * g * g / h;
}

double getVal(double g, double h) {
  return - g / h;
}

Tree::Tree() {}

void Tree::setNode(int node_id, int feature_id, double partition_value) {
  nodes[node_id].feature_id = feature_id;
  nodes[node_id].partition_value = partition_value;
}

TreeNode Tree::getNode(int node_id) {
  return nodes[node_id];
}

int Tree::addNode(int parent_id, int left_right, int nv) {
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

Forest::Forest(int tn, int td, int ln) {
  tree_num = tn;
  tree_depth = td;
  leave_num = ln;
}

int Forest::splitNode(NodePack& cur_nodepack, vector<ItemPack>& itempacks,
    const vector<DataItem>& data, int leaves) {
  if(leaves + 1 >= leave_num) {
    return -1;
  }
  int item_number = cur_nodepack.end_index - cur_nodepack.start_index;
  if(item_number == 0 || cur_nodepack.level >= MAX_DEPTH) { //interval with no item
    return -1;
  }

  double gains[FEATURE_NUMBER];
  int split_points[FEATURE_NUMBER];
  double left_values[FEATURE_NUMBER];
  double right_values[FEATURE_NUMBER];

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

  //find split feature with parallelization
  // #pragma omp parallel for
  for(int feature_id = 0; feature_id < FEATURE_NUMBER; feature_id++) {    
    //get features of each item
    vector<ItemWithOneFeature> index_feature(item_number);
    for(int i = 0; i < item_number; i++) {
      int itempack_index = cur_nodepack.start_index + i;
      index_feature[i].index = itempack_index;
      index_feature[i].feature = getFeature(data, itempacks[itempack_index].item_index, feature_id);
    }
    //sort items by feature value
    sort(index_feature.begin(), index_feature.end(), cmp);
    //caculate object function without split

    //find optimal split point
    double gl = 0.0, hl = 0.0, gr = G, hr = H;
    double opt_obj = ori_obj;
    double left_value, right_value;
    int split_point = -1;
    for(int i = 0; i < item_number - 1; i++) {
      int itempack_index = cur_nodepack.start_index + index_feature[i].index;
      gl += itempacks[itempack_index].first_order;
      hl += itempacks[itempack_index].second_order;
      gr -= itempacks[itempack_index].first_order;
      hr -= itempacks[itempack_index].second_order;
      if(getObj(gl, hl) + getObj(gr, hr) < opt_obj) {
        opt_obj = getObj(gl, hl) + getObj(gr, hr);
        split_point = itempack_index;
        left_value = getVal(gl, hl);
        right_value = getVal(gr, hr);
      }
    }
    gains[feature_id] = ori_obj - opt_obj;
    split_points[feature_id] = split_point;
    left_values[feature_id] = left_value;
    right_values[feature_id] = right_value;
  }

  //split
  int split_feature = -1;
  int split_point = -1;
  double gain, left_value, right_value;
  bool flag = 0;
  for(int feature_id = 0; feature_id < FEATURE_NUMBER; feature_id++) {
    // printf("feature id = %d ", feature_id);
    // printf("gain = %lf\n", gains[feature_id]);
    if(split_points[feature_id] != -1) {
      if(flag == 0) {
        flag = 1;
        split_feature = feature_id;
        split_point = split_points[feature_id];
        gain = gains[feature_id];
        // printf("gain = %lf\n", gain);
        // printf("feature id = %d\n", feature_id);
        left_value = left_values[feature_id];
        right_value = right_values[feature_id];
      }
      else if(gain < gains[feature_id]) {
        // printf("yes gain = %lf\n", gain);
        // printf("feature id = %d\n", feature_id);
        split_feature = feature_id;
        split_point = split_points[feature_id];
        gain = gains[feature_id];
        left_value = left_values[feature_id];
        right_value = right_values[feature_id];
      }
    }
  }
  if(split_point == -1) {//no suitable feature to split node
    return -1;
  }
  else {//split node
    rearrange(itempacks, cur_nodepack, split_feature,
        split_point, left_value, right_value, data);
    cur_nodepack.feature_id = split_feature;
    return split_point;
  }
}

void Forest::build(const vector<DataItem>& data, vector<int> indices) {
  trees.clear();
  //initialization: all items are assigned 0
  vector<ItemPack> itempacks;
  for(int i = 0; i < indices.size(); i++) {
    ItemPack itemPackToAdd;
    itemPackToAdd.item_index = indices[i];
    itemPackToAdd.current_sum = 0.0;
    itemPackToAdd.first_order = 2.0 * (- data[indices[i]].label);
    itemPackToAdd.second_order = 2.0;
    itempacks.push_back(itemPackToAdd);
  }

  //build trees one by one
  for(int tree_id = 0; tree_id < tree_num; tree_id++) {
    int leaves = 1;
    Tree treeToAdd;
    //add root
    int root_id = treeToAdd.addNode(-1, -1, 0.0); 
    vector<NodePack> nodepacks;
    NodePack rootNodePack;
    rootNodePack.node_id = root_id;
    rootNodePack.level = 0;
    rootNodePack.start_index = 0;
    rootNodePack.end_index = indices.size();
    rootNodePack.feature_id = -1; //not splited yet
    nodepacks.push_back(rootNodePack);

    int nodepack_index = 0;
    while(nodepack_index < nodepacks.size()) {
      NodePack cur_nodepack = nodepacks[nodepack_index];
      nodepack_index++;
      
      int split_point;
      split_point = splitNode(cur_nodepack, itempacks, data, leaves);
      nodepack_index++;
      if(split_point != -1) {
        leaves++;
        //update splited node
        double left_value = itempacks[cur_nodepack.start_index].current_value;
        double right_value = itempacks[cur_nodepack.end_index - 1].current_value;
        double partition_value = getFeature(data, itempacks[split_point].item_index, cur_nodepack.feature_id).value;
        treeToAdd.setNode(cur_nodepack.node_id, cur_nodepack.feature_id, partition_value);
        
        //add two new nodes
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

    //update sum, g & h
    for(int i = 0; i < itempacks.size(); i++) {
      itempacks[i].current_sum += itempacks[i].current_value;
      itempacks[i].first_order = 2.0 * (itempacks[i].current_sum - data[indices[i]].label);
      itempacks[i].second_order = 2.0;
    }

    trees.push_back(treeToAdd);
  }
}

void Tree::showTree() {
  assert(nodes.size() == left.size());
  assert(nodes.size() == right.size());
  for(int i = 0; i < nodes.size(); i++) {
    printf("node %d: left node = %d, right node = %d, feature id = %d, partition value = %lf\n",
        i, left[i], right[i], nodes[i].feature_id, nodes[i].partition_value);
  }
}

void Forest::showForest() {
  for(int i = 0; i < trees.size(); i++) {
    printf("tree %d:\n", i);
    trees[i].showTree();
    printf("\n");
  }
}