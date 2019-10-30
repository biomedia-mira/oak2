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

#include "oakForest.h"

namespace oak
{
  void train_tree(Node<Classifier>& root, const std::vector<DataSample<int>*>& dataSamples, const FeatureFactory& featureFactory, const std::vector<Feature>& featureSet, int countLabels, int maxDepth, int minSamples, bool improveParent)
  {
    // compute label weights used for data balancing
    auto label_weights = compute_label_weights(dataSamples, countLabels);

    std::cout << "+ label counts" << std::endl;
    for (int i = 0; i < countLabels; i++)
    {
      std::cout << "+ [" << i << "]: " << label_weights.second[i] << std::endl;
    }
    std::cout << "----------------------------------------" << std::endl;

    // initialize root node
    root.depth = 0;
    root.predictor = root_predictor(dataSamples, label_weights.first);

    std::queue<Node<Classifier>*> nodes;
    std::queue<std::vector<DataSample<int>*>> samples;
    nodes.push(&root);
    samples.push(dataSamples);

    int current_depth = 0;
    while (!nodes.empty())
    {
      auto node = nodes.front();

      if (node->depth < maxDepth)
      {
        if (current_depth == node->depth)
        {
          std::cout << "-- training " << nodes.size() << " node(s) on depth " << current_depth << std::endl;
          current_depth++;
        }

        train_node(*node, samples.front(), featureFactory, featureSet, countLabels, minSamples, improveParent, label_weights.first);

        if (!node->is_leaf)
        {
          nodes.push(node->left.get());
          nodes.push(node->right.get());

          auto splits = split_samples(samples.front(), node->split_function, featureFactory, featureSet);
          samples.push(splits.first);
          samples.push(splits.second);
        }
      }
      else
      {
        node->is_leaf = true;
      }

      nodes.pop();
      samples.pop();
    }
  }

  void train_node(Node<Classifier>& node, const std::vector<DataSample<int>*>& dataSamples, const FeatureFactory& featureFactory, const std::vector<Feature>& featureSet, int countLabels, int minSamples, bool improveParent, const Eigen::ArrayXd& labelWeights)
  {
    int count_features_per_node = featureFactory.count_features_per_node();
    double entropy_node = improveParent ? shannon_entropy(node.predictor.distribution) : std::numeric_limits<double>::max();

    // sample random feature subset
    auto feature_subset_indices = random_permutation(count_features_per_node, static_cast<int>(featureSet.size()));

    // iterate over all features and determine split candidates
    std::vector<SplitCandidate<Classifier>> split_candidates(count_features_per_node);

    // SIMPLE TBB
    tbb::parallel_for(0, count_features_per_node, [&](int f)
    {
      split_candidates[f] = split_candidate(dataSamples, feature_subset_indices[f], featureFactory, featureSet, countLabels, minSamples, labelWeights, entropy_node);
    });

    // ALTERNATIVE TBB WITH RANGES
    //tbb::parallel_for( tbb::blocked_range<size_t>(0,features.size()), [&](const tbb::blocked_range<size_t>& r)
    //{
    //  for(size_t i=r.begin(); i!=r.end(); ++i)
    //    split_candidates[i] = split_candidate(dataSamples, features[i], featureFactory, countLabels, minSamples, labelWeights, entropy_node);
    //});

    // WITHOUT TBB
    //for (int f = 0; f < features.size(); f++)
    //{
    //  split_candidates[f] = split_candidate(dataSamples, features[f], featureFactory, countLabels, minSamples, labelWeights, entropy_node);
    //}

    int count_valid_candidates = 0;
    for (int i = 0; i < split_candidates.size(); i++)
    {
      if (split_candidates[i].valid) count_valid_candidates++;
    }

    if (count_valid_candidates > 0)
    {
      SplitCandidate<Classifier> best_split;
      best_split.split_entropy = std::numeric_limits<double>::max();
      for (auto& split : split_candidates)
      {
        if (split.valid && split.split_entropy < best_split.split_entropy)
        {
          best_split = split;
        }
      }

      boost::shared_ptr<Node<Classifier>> node_left(new Node<Classifier>());
      boost::shared_ptr<Node<Classifier>> node_right(new Node<Classifier>());

      node.is_leaf = false;
      node.split_function = best_split.split_function;
      node.split_entropy = best_split.split_entropy;
      node.left = node_left;
      node.right = node_right;

      node_left->depth = node.depth + 1;
      node_left->predictor = best_split.predictor_left;

      node_right->depth = node.depth + 1;
      node_right->predictor = best_split.predictor_right;
    }
    else
    {
      node.is_leaf = true;
    }
  }

  SplitCandidate<Classifier> split_candidate(const std::vector<DataSample<int>*>& dataSamples, int featureIndex, const FeatureFactory& featureFactory, const std::vector<Feature>& featureSet, int countLabels, int minSamples, const Eigen::ArrayXd& labelWeights, double entropyParent)
  {
    int count_thresholds = featureFactory.count_thresholds();
    int stats_size = count_thresholds + 1;

    float min_response = std::numeric_limits<float>::max();
    float max_response = std::numeric_limits<float>::lowest();

    // compute feature responses for data samples
    std::vector<float> responses(dataSamples.size());
    for (int s = 0; s < dataSamples.size(); s++)
    {
      responses[s] = featureFactory.evaluate(featureSet[featureIndex], *(dataSamples[s]));
      if (responses[s] < min_response) min_response = responses[s];
      if (responses[s] > max_response) max_response = responses[s];
    }

    ClassifierStats stats(stats_size, countLabels);

    ThresholdHelper threshold_helper(min_response, max_response, count_thresholds);

    // initialize stats collection
    for (int s = 0; s < dataSamples.size(); s++)
    {
      int bin = threshold_helper.index(responses[s]);
      stats.increment(bin, dataSamples[s]->label);
    }

    size_t count_samples_total = stats.count_samples(0);
    Eigen::ArrayXd label_counts_total(stats.label_counts(0));
    for (int bin = 1; bin < stats_size; bin++)
    {
      count_samples_total += stats.count_samples(bin);
      label_counts_total += stats.label_counts(bin);
    }

    size_t count_samples_left = 0;
    Eigen::ArrayXd label_counts_left(countLabels);
    label_counts_left.setZero();

    size_t count_samples_right = count_samples_total;
    Eigen::ArrayXd label_counts_right(label_counts_total);

    SplitCandidate<Classifier> best_split_candidate;
    best_split_candidate.split_function.feature_index = featureIndex;
    double best_split_entropy = entropyParent;
    for (int thresh_index = 0; thresh_index < count_thresholds; thresh_index++)
    {
      count_samples_left += stats.count_samples(thresh_index);
      count_samples_right -= stats.count_samples(thresh_index);
      label_counts_left += stats.label_counts(thresh_index);
      label_counts_right -= stats.label_counts(thresh_index);

      if (count_samples_left > minSamples && count_samples_right  > minSamples)
      {
        std::pair<Eigen::ArrayXd, double> distribution_left = compute_class_distribution(label_counts_left, labelWeights);
        std::pair<Eigen::ArrayXd, double> distribution_right = compute_class_distribution(label_counts_right, labelWeights);

        Classifier predictor_left(distribution_left.first, count_samples_left);
        Classifier predictor_right(distribution_right.first, count_samples_right);

        double entropy_left = shannon_entropy(distribution_left.first);

        double entropy_right = shannon_entropy(distribution_right.first);

        double split_entropy = (entropy_left * distribution_left.second + entropy_right * distribution_right.second) / (distribution_left.second + distribution_right.second);

        if (split_entropy < best_split_entropy)
        {
          best_split_entropy = split_entropy;
          best_split_candidate.split_entropy = split_entropy;
          best_split_candidate.predictor_left = predictor_left;
          best_split_candidate.predictor_right = predictor_right;
          best_split_candidate.split_function.threshold = threshold_helper.value(thresh_index);
        }
      }
    }

    if (best_split_entropy < entropyParent)
    {
      best_split_candidate.valid = true;
    }
    else
    {
      best_split_candidate.valid = false;
    }

    return best_split_candidate;
  }

  void train_tree(Node<Regressor>& root, const std::vector<DataSample<Eigen::VectorXd>*>& dataSamples, const FeatureFactory& featureFactory, const std::vector<Feature>& featureSet, int countVariables, int maxDepth, int minSamples, bool improveParent)
  {
    // initialize root node
    root.depth = 0;
    root.predictor = root_predictor(dataSamples, countVariables);

    std::queue<Node<Regressor>*> nodes;
    std::queue<std::vector<DataSample<Eigen::VectorXd>*>> samples;
    nodes.push(&root);
    samples.push(dataSamples);

    int current_depth = 0;
    while (!nodes.empty())
    {
      auto node = nodes.front();

      if (node->depth < maxDepth)
      {
        if (current_depth == node->depth)
        {
          std::cout << "-- training " << nodes.size() << " node(s) on depth " << current_depth << std::endl;
          current_depth++;
        }

        train_node(*node, samples.front(), featureFactory, featureSet, countVariables, minSamples, improveParent);

        if (!node->is_leaf)
        {
          nodes.push(node->left.get());
          nodes.push(node->right.get());

          auto splits = split_samples(samples.front(), node->split_function, featureFactory, featureSet);
          samples.push(splits.first);
          samples.push(splits.second);
        }
      }
      else
      {
        node->is_leaf = true;
      }

      nodes.pop();
      samples.pop();
    }
  }

  void train_node(Node<Regressor>& node, const std::vector<DataSample<Eigen::VectorXd>*>& dataSamples, const FeatureFactory& featureFactory, const std::vector<Feature>& featureSet, int countVariables, int minSamples, bool improveParent)
  {
    int count_features_per_node = featureFactory.count_features_per_node();
    double entropy_node = improveParent ? differential_entropy(node.predictor.variances) : std::numeric_limits<double>::max();

    // sample random feature subset
    auto feature_subset_indices = random_permutation(count_features_per_node, static_cast<int>(featureSet.size()));

    // iterate over all features and determine split candidates
    std::vector<SplitCandidate<Regressor>> split_candidates(count_features_per_node);

    // SIMPLE TBB
    tbb::parallel_for(0, count_features_per_node, [&](int f)
    {
      split_candidates[f] = split_candidate(dataSamples, feature_subset_indices[f], featureFactory, featureSet, countVariables, minSamples, entropy_node);
    });

    // ALTERNATIVE TBB WITH RANGES
    //tbb::parallel_for( tbb::blocked_range<size_t>(0,features.size()), [&](const tbb::blocked_range<size_t>& r)
    //{
    //  for(size_t i=r.begin(); i!=r.end(); ++i)
    //    split_candidates[i] = split_candidate(dataSamples, features[i], featureFactory, countLabels, minSamples, labelWeights, entropy_node);
    //});

    // WITHOUT TBB
    //for (int f = 0; f < features.size(); f++)
    //{
    //  split_candidates[f] = split_candidate(dataSamples, features[f], featureFactory, countLabels, minSamples, labelWeights, entropy_node);
    //}

    int count_valid_candidates = 0;
    for (int i = 0; i < split_candidates.size(); i++)
    {
      if (split_candidates[i].valid) count_valid_candidates++;
    }

    if (count_valid_candidates > 0)
    {
      SplitCandidate<Regressor> best_split;
      best_split.split_entropy = std::numeric_limits<double>::max();
      for (auto& split : split_candidates)
      {
        if (split.valid && split.split_entropy < best_split.split_entropy)
        {
          best_split = split;
        }
      }

      boost::shared_ptr<Node<Regressor>> node_left(new Node<Regressor>());
      boost::shared_ptr<Node<Regressor>> node_right(new Node<Regressor>());

      node.is_leaf = false;
      node.split_function = best_split.split_function;
      node.split_entropy = best_split.split_entropy;
      node.left = node_left;
      node.right = node_right;

      node_left->depth = node.depth + 1;
      node_left->predictor = best_split.predictor_left;

      node_right->depth = node.depth + 1;
      node_right->predictor = best_split.predictor_right;
    }
    else
    {
      node.is_leaf = true;
    }
  }

  SplitCandidate<Regressor> split_candidate(const std::vector<DataSample<Eigen::VectorXd>*>& dataSamples, int featureIndex, const FeatureFactory& featureFactory, const std::vector<Feature>& featureSet, int countVariables, int minSamples, double entropyParent)
  {
    int count_thresholds = featureFactory.count_thresholds();
    int stats_size = count_thresholds + 1;

    float min_response = std::numeric_limits<float>::max();
    float max_response = std::numeric_limits<float>::lowest();

    // compute feature responses for data samples
    std::vector<float> responses(dataSamples.size());
    for (int s = 0; s < dataSamples.size(); s++)
    {
      responses[s] = featureFactory.evaluate(featureSet[featureIndex], *(dataSamples[s]));
      if (responses[s] < min_response) min_response = responses[s];
      if (responses[s] > max_response) max_response = responses[s];
    }

    RegressorStats stats(stats_size, countVariables);

    ThresholdHelper threshold_helper(min_response, max_response, count_thresholds);

    // initialize stats collection
    for (int s = 0; s < dataSamples.size(); s++)
    {
      int bin = threshold_helper.index(responses[s]);
      stats.increment(bin, dataSamples[s]->label);
    }

    size_t count_samples_total = stats.count_samples(0);
    Eigen::ArrayXd sums_total(stats.sums(0));
    Eigen::ArrayXd sums_of_squares_total(stats.sums_of_squares(0));
    for (int bin = 1; bin < stats_size; bin++)
    {
      count_samples_total += stats.count_samples(bin);
      sums_total += stats.sums(bin);
      sums_of_squares_total += stats.sums_of_squares(bin);
    }

    size_t count_samples_left = 0;
    Eigen::ArrayXd sums_left(countVariables);
    Eigen::ArrayXd sums_of_squares_left(countVariables);
    sums_left.setZero();
    sums_of_squares_left.setZero();

    size_t count_samples_right = count_samples_total;
    Eigen::ArrayXd sums_right(sums_total);
    Eigen::ArrayXd sums_of_squares_right(sums_of_squares_total);

    SplitCandidate<Regressor> best_split_candidate;
    best_split_candidate.split_function.feature_index = featureIndex;
    double best_split_entropy = entropyParent;
    //double best_split_entropy = std::numeric_limits<double>::max();
    for (int thresh_index = 0; thresh_index < count_thresholds; thresh_index++)
    {
      count_samples_left += stats.count_samples(thresh_index);
      count_samples_right -= stats.count_samples(thresh_index);
      sums_left += stats.sums(thresh_index);
      sums_right -= stats.sums(thresh_index);
      sums_of_squares_left += stats.sums_of_squares(thresh_index);
      sums_of_squares_right -= stats.sums_of_squares(thresh_index);

      if (count_samples_left > minSamples && count_samples_right  > minSamples)
      {
        std::pair<Eigen::ArrayXd, Eigen::ArrayXd> means_and_variances_left = compute_means_and_variances(sums_left, sums_of_squares_left, count_samples_left);
        std::pair<Eigen::ArrayXd, Eigen::ArrayXd> means_and_variances_right = compute_means_and_variances(sums_right, sums_of_squares_right, count_samples_right);

        Regressor predictor_left(means_and_variances_left.first, means_and_variances_left.second, count_samples_left);
        Regressor predictor_right(means_and_variances_right.first, means_and_variances_right.second, count_samples_right);

        double entropy_left = differential_entropy(means_and_variances_left.second);

        double entropy_right = differential_entropy(means_and_variances_right.second);

        double split_entropy = (entropy_left * static_cast<double>(count_samples_left)+entropy_right * static_cast<double>(count_samples_right)) / static_cast<double>(count_samples_total);

        if (split_entropy < best_split_entropy)
        {
          best_split_entropy = split_entropy;
          best_split_candidate.split_entropy = split_entropy;
          best_split_candidate.predictor_left = predictor_left;
          best_split_candidate.predictor_right = predictor_right;
          best_split_candidate.split_function.threshold = threshold_helper.value(thresh_index);
        }
      }
    }

    if (best_split_entropy < entropyParent)
    {
      best_split_candidate.valid = true;
    }
    else
    {
      best_split_candidate.valid = false;
    }

    return best_split_candidate;
  }
}
