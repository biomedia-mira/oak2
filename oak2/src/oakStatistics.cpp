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

#include "oakStatistics.h"

namespace oak
{
  ClassifierStats::ClassifierStats(size_t size, int countLabels)
    : m_label_counts(size)
    , m_count_samples(size)
  {
    for (auto& h : m_label_counts)
      h = Eigen::ArrayXd(countLabels);

    reset();
  }

  void ClassifierStats::reset()
  {
    for (auto& c : m_label_counts)
      c.setZero();
    for (auto& c : m_count_samples)
      c = 0;
  }

  void ClassifierStats::increment(int index, int label)
  {
    m_label_counts[index][label]++;
    m_count_samples[index]++;
  }

  std::pair<Eigen::ArrayXd, Eigen::ArrayXd> compute_label_weights(const std::vector<DataSample<int>*>& dataSamples, int countLabels)
  {
    Eigen::ArrayXd label_counts = Eigen::ArrayXd(countLabels);
    label_counts.setZero();
    for (auto& sample : dataSamples)
      label_counts[sample->label]++;

    double sum = static_cast<double>(dataSamples.size());

    Eigen::ArrayXd label_weights = Eigen::ArrayXd(countLabels);
    for (int i = 0; i < label_weights.size(); i++)
    {
      label_weights[i] = label_counts[i] > 0 ? sum / label_counts[i] : 0;
    }
    return std::make_pair(label_weights, label_counts);
  }

  std::pair<Eigen::ArrayXd, double> compute_class_distribution(const Eigen::ArrayXd& labelCounts, const Eigen::ArrayXd& labelWeights)
  {
    Eigen::ArrayXd distribution = labelCounts.cwiseProduct(labelWeights);
    double sum = distribution.sum();
    distribution /= sum;

    return std::make_pair(distribution, sum);
  }

  Classifier root_predictor(const std::vector<DataSample<int>*>& dataSamples, const Eigen::ArrayXd& labelWeights)
  {
    Eigen::ArrayXd label_counts(labelWeights);
    label_counts.setZero();
    for (auto& sample : dataSamples)
      label_counts[sample->label]++;

    std::pair<Eigen::ArrayXd, double> distribution = compute_class_distribution(label_counts, labelWeights);

    return Classifier(distribution.first, dataSamples.size());
  }

  void update_predictor(const std::vector<DataSample<int>*>& dataSamples, Classifier& predictor)
  {
    auto count_samples = predictor.count_samples;
    float scale = static_cast<float>(count_samples) / static_cast<float>(count_samples + dataSamples.size());

    for (int i = 0; i < predictor.distribution.size(); i++)
    {
      predictor.distribution[i] *= scale;
    }

    for (auto& sample : dataSamples)
      predictor.distribution[sample->label] += scale;

    predictor.count_samples += dataSamples.size();
  }

  double shannon_entropy(const Eigen::ArrayXd& distribution)
  {
    double entropy = 0;
    for (int i = 0; i < distribution.size(); i++)
    {
      double prob = distribution[i];
      if (prob > 0) entropy -= prob * log(prob);
    }
    return entropy;
  }

  RegressorStats::RegressorStats(size_t size, int countVariables)
    : m_sums(size)
    , m_sums_of_squares(size)
    , m_count_samples(size)
  {
    for (auto& h : m_sums)
      h = Eigen::ArrayXd(countVariables);
    for (auto& h : m_sums_of_squares)
      h = Eigen::ArrayXd(countVariables);

    reset();
  }

  void RegressorStats::reset()
  {
    for (auto& c : m_sums)
      c.setZero();
    for (auto& c : m_sums_of_squares)
      c.setZero();
    for (auto& c : m_count_samples)
      c = 0;
  }

  void RegressorStats::increment(int index, Eigen::VectorXd label)
  {
    for (int i = 0; i < label.size(); i++)
    {
      m_sums[index][i] += label[i];
      m_sums_of_squares[index][i] += label[i] * label[i];
    }
    m_count_samples[index]++;
  }

  std::pair<Eigen::ArrayXd, Eigen::ArrayXd> compute_means_and_variances(const Eigen::ArrayXd& sums, const Eigen::ArrayXd& sumsOfSquares, size_t countSamples)
  {
    Eigen::ArrayXd means = sums / static_cast<double>(countSamples);
    Eigen::ArrayXd variances = sumsOfSquares / static_cast<double>(countSamples)-means.cwiseProduct(means);

    return std::make_pair(means, variances);
  }

  Regressor root_predictor(const std::vector<DataSample<Eigen::VectorXd>*>& dataSamples, int countVariables)
  {
    Eigen::ArrayXd sums(countVariables);
    Eigen::ArrayXd sums_of_squares(countVariables);
    sums.setZero();
    sums_of_squares.setZero();
    for (auto& sample : dataSamples)
    {
      Eigen::VectorXd label = sample->label;
      for (int i = 0; i < label.size(); i++)
      {
        sums[i] += label[i];
        sums_of_squares[i] += label[i] * label[i];
      }
    }

    std::pair<Eigen::ArrayXd, Eigen::ArrayXd> means_and_variances = compute_means_and_variances(sums, sums_of_squares, dataSamples.size());

    return Regressor(means_and_variances.first, means_and_variances.second, dataSamples.size());
  }

  void update_predictor(const std::vector<DataSample<Eigen::VectorXd>*>& dataSamples, Regressor& predictor)
  {
    for (auto& sample : dataSamples)
    {
      Eigen::VectorXd label = sample->label;
      for (int i = 0; i < label.size(); i++)
      {
        auto n = static_cast<double>(predictor.count_samples + 1);
        auto old_mean = predictor.means[i];
        auto new_mean = old_mean + (label[i] - predictor.means[i]) / n;
        predictor.means[i] = new_mean;

        auto old_var = predictor.variances[i];
        auto new_var = ((n - 1.0)*old_var + (label[i] - old_mean) * (label[i] - new_mean)) / n;
        predictor.variances[i] = new_var;
        predictor.count_samples = static_cast<size_t>(n);
      }
    }
  }

  double differential_entropy(const Eigen::ArrayXd& variances)
  {
    return variances.sum() + 1.0;
  }

  ThresholdHelper::ThresholdHelper(float minResponse, float maxResponse, int countThresholds)
  {
    m_min_threshold = minResponse + std::numeric_limits<float>::epsilon();
    m_max_threshold = maxResponse;
    if (countThresholds > 1)
    {
      m_bin_width = (m_max_threshold - m_min_threshold) / static_cast<float>(countThresholds - 1);
    }
    else
    {
      m_bin_width = 0;
      m_min_threshold = (maxResponse - minResponse) / 2 + minResponse;
      m_max_threshold = m_min_threshold;
    }

    m_count_thresholds = countThresholds;
  }

  int ThresholdHelper::index(float response)
  {
    //int index = static_cast<int>((response - min_response) / (max_response - min_response + 1.0f) * static_cast<float>(m_count_thresholds+1));
    //if (index == m_count_thresholds+1) index--;

    if (response < m_min_threshold)
    {
      return 0;
    }
    else if (response >= m_max_threshold)
    {
      return m_count_thresholds;
    }
    else
    {
      int index = static_cast<int>((response - m_min_threshold) / m_bin_width) + 1;
      return index >= m_count_thresholds ? m_count_thresholds - 1 : index;
    }
  }

  float ThresholdHelper::value(int index)
  {
    //return min_response + (max_response - min_response + 1.0f) / static_cast<float>(count_thresholds+1) * static_cast<float>(index+1);

    if (index <= 0)
    {
      return m_min_threshold;
    }
    else if (index >= m_count_thresholds - 1)
    {
      return m_max_threshold;
    }
    else
    {
      return m_min_threshold + static_cast<float>(index)* m_bin_width;
    }
  }

  std::vector<int> random_permutation(int m, int n)
  {
    std::vector<int> indices(n);
    for (int i = 0; i < n; i++)
    {
      indices[i] = i;
    }

    std::random_device rd;
    std::mt19937 mt(rd());

    for (int i = 0; i < m; i++)
    {
      std::uniform_int_distribution<int> uid(i, n - 1);
      int r = uid(mt);
      std::swap(indices[i], indices[r]);
    }

    return indices;
  }
}
