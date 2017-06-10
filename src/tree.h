#ifndef __TREE_H__
#define __TREE_H__

#include "data.h"
#include <omp.h>

struct ItemPack {
  int item_index; //denote the index in dataset
  double current_value;
  double current_sum;
  double first_order;
  double second_order;
};

struct NodePack {
  int node_id;
  int level;
  int feature_id;
  int start_index;
  int end_index;
};

struct TreeNode {
  int level;
  int feature_id; //the id of feature that splits this node
  double partition_value; //the value of feature that splits this node
  double node_value; //the value of item that is assigned to this node
};

class Tree {
 public:
  Tree();
  void setNode(int node_id, int feature_id, double partition_value);
  TreeNode getNode(int node_id);
  int addNode(int parent_id, int left_right, int nv);
 
 private:
  vector<TreeNode> nodes;
  vector<int> left; //the node_id of left spring
  vector<int> right; //the node_id of right spring
};

class Forest {
 public:
  Forest(int tn, int td, int ln);
  void build(const vector<DataItem>& data, vector<int> indices);
 
 private:
  int tree_num;
  int tree_depth;
  int leave_num;
  vector<Tree> trees;

  //return split point in itempacks
  //return -1 if not split
  int splitNode(NodePack& cur_nodepack, 
    vector<ItemPack>& itempacks, const vector<DataItem>& data);
};

#endif
