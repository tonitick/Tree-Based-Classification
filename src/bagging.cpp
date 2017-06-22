#include "bagging.h"

#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

int random(int start, int end) {
  return (int)((double)(start) + (double)(end - start) * rand() / (RAND_MAX + 1.0));
}

void Bagging::scatter(vector<int>& input) {
  int intput_size = input.size();
  for(int i = 0; i < SCATTER_RATIO * intput_size; i++) {
    int a = random(0, intput_size);
    int b = random(0, intput_size);
    swap(input, a, b);
  }
}

void Bagging::swap(vector<int>& input, int i, int j) {
  int tmp = input[i];
  input[i] = input[j];
  input[j] = tmp; 
}

Bagging::Bagging(int data_size, int feature_size) {
  bagging_size = 0;
  losses.clear();
  gbdts.clear();
  item_candidate.clear();
  feature_cadidate.clear();
  for(int i = 0; i < data_size; i++) {
    item_candidate.push_back(i);
  }
  for(int i = 1; i <= feature_size; i++) {
    feature_cadidate.push_back(i);
  }
}

void Bagging::addOne(const vector<DataItem>& data, int item_num, int feature_num) {
  //generate data items and features randomly
  scatter(item_candidate);
  scatter(feature_cadidate);
  vector<int> items;
  for(int i = 0; i < item_num; i++) {
    items.push_back(item_candidate[i]);
  }
  vector<int> features;
  for(int i = 0; i < feature_num; i++) {
    features.push_back(feature_cadidate[i]);
  }

  //build gbdt
  GBDT gbdt(BOOSTING_SIZE, MAX_DEPTH, MAX_LEAVES); //to add
  gbdts.push_back(gbdt);
  double loss = gbdts[bagging_size].build(data, items, features);
  losses.push_back(loss);

  bagging_size++;
}

vector<double> Bagging::estimate(const vector<DataItem>& data) {
  //get weight
  vector<double> weight(bagging_size);
  double sum = 0.0;
  assert(bagging_size == losses.size());
  for(int i = 0; i < bagging_size; i++) {
    sum += 1.0 / losses[i];
  }
  for(int i = 0; i < bagging_size; i++) {
    weight[i] = 1.0 / losses[i] / sum;
  }

  //get result
  vector<double> result(data.size());
  for(int i = 0; i < result.size(); i++) {
    assert(result[i] == 0.0);
  }
  assert(bagging_size == gbdts.size());
  for(int i = 0; i < bagging_size; i++) {
    vector<double> sub_result = gbdts[i].estimate(data);
    for(int j = 0; j < data.size(); j++) {
      result[j] += weight[i] * sub_result[j];
    }
  }

  return result;
}

void Bagging::crossValidation(const vector<DataItem>& data, int val_size) {
  scatter(item_candidate);

  vector<DataItem> val_data;
  for(int i = 0; i < val_size; i++) {
    val_data.push_back(data[item_candidate[i]]);
  }

  //get weight
  vector<double> weight(bagging_size);
  double sum = 0.0;
  assert(bagging_size == losses.size());
  for(int i = 0; i < bagging_size; i++) {
    sum += 1.0 / losses[i];
  }
  for(int i = 0; i < bagging_size; i++) {
    weight[i] = 1.0 / losses[i] / sum;
    // printf("bag %d: loss = %lf, weight = %lf\n", losses[i], weight[i]);
  }

  //get result
  vector<double> result(val_size);
  for(int i = 0; i < val_size; i++) {
    assert(result[i] == 0.0);
  }
  assert(bagging_size == gbdts.size());
  for(int i = 0; i < bagging_size; i++) {
    vector<double> sub_result = gbdts[i].estimate(val_data);
    for(int j = 0; j < val_size; j++) {
      // printf("weight = %lf, sub result = %lf\n", weight[i], sub_result[j]);
      result[j] += weight[i] * sub_result[j];
    }
  }

  double loss = 0.0;
  for(int i = 0; i < val_size; i++) {
    loss += (result[i] - data[item_candidate[i]].label) * (result[i] - data[item_candidate[i]].label);
    // printf("result i = %lf\n", result[i]);
  }
  printf("average loss of cross validation: %lf\n", loss / val_size);
}