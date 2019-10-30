/*
* oak2 - a random forest library for 3D image analysis.
*
* Copyright 2016 Ben Glocker <b.glocker@imperial.ac.uk>
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include <queue>
#include <iostream>
#include <tbb/parallel_for.h>

namespace oak
{
  template<typename label_type>
  std::pair<std::vector<DataSample<label_type>*>, std::vector<DataSample<label_type>*>> split_samples(const std::vector<DataSample<label_type>*>& dataSamples, const SplitFunction& function, const FeatureFactory& featureFactory, const std::vector<Feature>& featureSet)
  {
    std::vector<DataSample<label_type>*> samples_left;
    std::vector<DataSample<label_type>*> samples_right;

    // SIMPLE TBB
    size_t count_samples = dataSamples.size();
    std::vector<float> results(count_samples);
    tbb::parallel_for<size_t>(0, count_samples, [&](size_t i)
    {
      results[i] = evaluate_splitfunction(function, *dataSamples[i], featureFactory, featureSet);
    });

    for (size_t i = 0; i < count_samples; i++)
    {
      if (results[i] >= 0.0f)
        samples_left.push_back(dataSamples[i]);
      else
        samples_right.push_back(dataSamples[i]);
    }

    // WITHOUT TBB
    //for (auto& sample : dataSamples)
    //{
    //  float res = evaluate_splitfunction(function, *sample, featureFactory);
    //
    //  if (res >= 0.0f)
    //    samples_left.push_back(sample);
    //  else
    //    samples_right.push_back(sample);
    //}

    return std::make_pair(samples_left, samples_right);
  }

  template<typename label_type, typename predictor_type>
  const Node<predictor_type>& descend_tree(const Node<predictor_type>& root, const DataSample<label_type>& sample, const FeatureFactory& featureFactory, const std::vector<Feature>& featureSet, int maxDepth)
  {
    const Node<predictor_type>* node = &root;
    while(!node->is_leaf && node->depth < maxDepth)
    {
      float res = evaluate_splitfunction(node->split_function, sample, featureFactory, featureSet);

      if (res >= 0.0f)
        node = node->left.get();
      else
        node = node->right.get();

    }
    return *node;
  }

  template<typename label_type, typename predictor_type>
  int node_index(const Node<predictor_type>& root, const DataSample<label_type>& sample, const FeatureFactory& featureFactory, const std::vector<Feature>& featureSet, int maxDepth)
  {
    int nodeIndex = 0;
    const Node<predictor_type>* node = &root;
    while (!node->is_leaf && node->depth < maxDepth)
    {
      float res = evaluate_splitfunction(node->split_function, sample, featureFactory, featureSet);

      int bit;
      if (res >= 0.0f)
      {
        node = node->left.get();
        bit = 0;
      }
      else
      {
        node = node->right.get();
        bit = 1;
      }

      nodeIndex += bit * pow(2, node->depth);
    }

    nodeIndex += pow(2, node->depth) - 1;

    return nodeIndex;
  }

  template<typename label_type, typename predictor_type>
  std::pair<const Node<predictor_type>&, int> descend_tree_node_index(const Node<predictor_type>& root, const DataSample<label_type>& sample, const FeatureFactory& featureFactory, const std::vector<Feature>& featureSet, int maxDepth)
  {
    int nodeIndex = 0;
    const Node<predictor_type>* node = &root;
    while (!node->is_leaf && node->depth < maxDepth)
    {
      float res = evaluate_splitfunction(node->split_function, sample, featureFactory, featureSet);

      int bit;
      if (res >= 0.0f)
      {
        node = node->left.get();
        bit = 0;
      }
      else
      {
        node = node->right.get();
        bit = 1;
      }

      nodeIndex += bit * pow(2, node->depth);
    }

    nodeIndex += pow(2, node->depth) - 1;

    return std::make_pair(*node, nodeIndex);
  }

  template<typename label_type, typename predictor_type>
  std::pair<std::vector<const Node<predictor_type>*>, int> descend_tree_path_node_index(const Node<predictor_type>& root, const DataSample<label_type>& sample, const FeatureFactory& featureFactory, const std::vector<Feature>& featureSet, int maxDepth)
  {
    int nodeIndex = 0;
    const Node<predictor_type>* node = &root;
    std::vector<const Node<predictor_type>*> path;
    path.push_back(node);
    while (!node->is_leaf && node->depth < maxDepth)
    {
      float res = evaluate_splitfunction(node->split_function, sample, featureFactory, featureSet);

      int bit;
      if (res >= 0.0f)
      {
        node = node->left.get();
        bit = 0;
      }
      else
      {
        node = node->right.get();
        bit = 1;
      }

      path.push_back(node);

      nodeIndex += bit * pow(2, node->depth);
    }

    nodeIndex += pow(2, node->depth) - 1;

    return std::make_pair(path, nodeIndex);
  }

  template<typename label_type, typename predictor_type>
  void update_tree(Node<predictor_type>& root, const std::vector<DataSample<label_type>*>& dataSamples, const FeatureFactory& featureFactory, const std::vector<Feature>& featureSet)
  {
    std::queue<Node<predictor_type>*> nodes;
    std::queue<std::vector<DataSample<label_type>*>> samples;
    nodes.push(&root);
    samples.push(dataSamples);

    int current_depth = 0;
    while(!nodes.empty())
    {
      auto node = nodes.front();
      update_predictor(samples.front(), node->predictor);

      if (!node->is_leaf)
      {
        nodes.push(node->left.get());
        nodes.push(node->right.get());

        auto splits = split_samples(samples.front(), node->split_function, featureFactory, featureSet);
        samples.push(splits.first);
        samples.push(splits.second);
      }

      nodes.pop();
      samples.pop();
    }
  }
}
