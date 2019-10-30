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

#pragma once

#include "oakFeatures.h"
#include <vector>

namespace oak
{
//##################
//  CLASSIFICATION
//##################

  struct Classifier
  {
    Classifier()
      : distribution()
      , count_samples(0)
    {}

    Classifier(Eigen::ArrayXd distribution, size_t countSamples)
      : distribution(distribution)
      , count_samples(countSamples)
    {}

    size_t count_samples;
    Eigen::ArrayXd distribution;

    /**
     * \brief Serializes objects of type oak::Classifier.
     **/
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
      ar & count_samples;
      ar & distribution;
    }
  };

  class ClassifierStats
  {
  public:
    ClassifierStats(size_t size, int countLabels);

    /**
     * \brief Resets sample and label counts.
     **/
    void reset();

    /**
     * \brief Increments the counts for the given label on the specified stats index.
     * \param index The stats index.
     * \param label The label.
     **/
    void increment(int index, int label);

    /**
     * \brief Gets the label count for the specified stats index.
     * \param index The stats index.
     * \return The label counts.
     **/
    const Eigen::ArrayXd& label_counts(int index) const
    {
      return m_label_counts[index];
    }

    /**
     * \brief Gets the sample count for the specified stats index.
     * \param index The stats index.
     * \return The sample count.
     **/
    size_t count_samples(int index)
    {
      return m_count_samples[index];
    }

  private:
    std::vector<Eigen::ArrayXd> m_label_counts;
    std::vector<size_t> m_count_samples;
  };

  /**
   * \brief Computes the label weights used to compensate for unbalanced data sampling.
   * \param dataSamples The data samples.
   * \param countLabels The number of categorical labels.
   * \return The label weights and counts.
   **/
  std::pair<Eigen::ArrayXd,Eigen::ArrayXd> compute_label_weights(const std::vector<DataSample<int>*>& dataSamples, int countLabels);

  /**
   * \brief Computes the empirical probability distribution over categorical labels.
   * \param labelCounts The label counts.
   * \param labelWeights The label weights.
   * \return The probability distribution.
   **/
  std::pair<Eigen::ArrayXd, double> compute_class_distribution(const Eigen::ArrayXd& labelCounts, const Eigen::ArrayXd& labelWeights);

  /**
   * \brief Determines the root node predictor.
   * \param dataSamples The data samples.
   * \param labelWeights The label weights.
   * \return The predictor.
   **/
  Classifier root_predictor(const std::vector<DataSample<int>*>& dataSamples, const Eigen::ArrayXd& labelWeights);

  /**
   * \brief Updates a given predictor with new data samples.
   * \param dataSamples The data samples.
   * \param predictor The predictor.
   **/
  void update_predictor(const std::vector<DataSample<int>*>& dataSamples, Classifier& predictor);

  /**
   * \brief Computes the Shannon entropy for a given label probability distribution.
   * \param distribution The probability distribution.
   * \return The entropy.
   **/
  double shannon_entropy(const Eigen::ArrayXd& distribution);

//##################
//  REGRESSION
//##################

  struct Regressor
  {
    Regressor()
      : means()
      , variances()
      , count_samples(0)
    {}

    Regressor(Eigen::ArrayXd means, Eigen::ArrayXd variances, size_t countSamples)
      : means(means)
      , variances(variances)
      , count_samples(countSamples)
    {}

    size_t count_samples;
    Eigen::ArrayXd means;
    Eigen::ArrayXd variances;

    /**
     * \brief Serializes objects of type oak::Regressor.
     **/
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
      ar & count_samples;
      ar & means;
      ar & variances;
    }
  };

  class RegressorStats
  {
  public:
    RegressorStats(size_t size, int countVariables);

    /**
     * \brief Resets sample counts and sums.
     **/
    void reset();

    /**
     * \brief Updates the sums for the given label on the specified stats index.
     * \param index The stats index.
     * \param label The label.
     **/
    void increment(int index, Eigen::VectorXd label);

    /**
     * \brief Gets the sums for the specified stats index.
     * \param index The stats index.
     * \return The sums.
     **/
    const Eigen::ArrayXd& sums(int index) const
    {
      return m_sums[index];
    }

    /**
     * \brief Gets the sums of squares for the specified stats index.
     * \param index The stats index.
     * \return The sums of squares.
     **/
    const Eigen::ArrayXd& sums_of_squares(int index) const
    {
      return m_sums_of_squares[index];
    }

    /**
     * \brief Gets the sample count for the specified stats index.
     * \param index The stats index.
     * \return The sample count.
     **/
    size_t count_samples(int index)
    {
      return m_count_samples[index];
    }

  private:
    std::vector<Eigen::ArrayXd> m_sums;
    std::vector<Eigen::ArrayXd> m_sums_of_squares;
    std::vector<size_t> m_count_samples;
  };

  /**
   * \brief Computes the empirical means and variances over multi-valued, continuous labels.
   * \param sums The sums over labels.
   * \param sumsOfSquares The sums of squares over labels.
   * \param countSamples The sample count.
   * \return The means and variances.
   **/
  std::pair<Eigen::ArrayXd, Eigen::ArrayXd> compute_means_and_variances(const Eigen::ArrayXd& sums, const Eigen::ArrayXd& sumsOfSquares, size_t countSamples);

  /**
   * \brief Determines the root node predictor.
   * \param dataSamples The data samples.
   * \param countVariables The number of variables.
   * \return The predictor.
   **/
  Regressor root_predictor(const std::vector<DataSample<Eigen::VectorXd>*>& dataSamples, int countVariables);

  /**
   * \brief Updates a given predictor with new data samples.
   * \param dataSamples The data samples.
   * \param predictor The predictor.
   **/
  void update_predictor(const std::vector<DataSample<Eigen::VectorXd>*>& dataSamples, Regressor& predictor);

  /**
   * \brief Computes the differential entropy for a given set of variances.
   * \param variances The variances.
   * \return The entropy.
   **/
  double differential_entropy(const Eigen::ArrayXd& variances);

//##################
//  GENERAL
//##################

  struct SplitFunction
  {
    int feature_index;
    float threshold;

    /**
     * \brief Serializes objects of type oak::SplitFunction.
     **/
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
      ar & feature_index;
      ar & threshold;
    }
  };

  template<typename predictor_type>
  struct SplitCandidate
  {
    SplitFunction split_function;
    predictor_type predictor_left;
    predictor_type predictor_right;
    double split_entropy;
    bool valid;
  };

  class ThresholdHelper
  {
  public:
    ThresholdHelper(float minResponse, float maxResponse, int countThresholds);

    /**
     * \brief Gets the threshold index for a given feature response.
     * \param response The feature response.
     * \return The threshold index.
     **/
    int index(float response);

    /**
     * \brief Gets the threshold value for a given index.
     * \param index The threshold index.
     * \return The threshold value.
     **/
    float value(int index);

  private:
    float m_min_threshold;
    float m_max_threshold;
    float m_bin_width;
    int m_count_thresholds;
  };

  /**
   * \brief Evaluates a split function on a given data sample.
   * \param function The split function.
   * \param sample The data sample.
   * \param factory The feature factory.
   * \param featureSet The feature set.
   * \return The signed distance of the features response to the threshold value.
   **/
  template<typename label_type>
  float evaluate_splitfunction(const SplitFunction& function, const DataSample<label_type>& sample, const FeatureFactory& factory, const std::vector<Feature>& featureSet);

  /**
   * \brief Computes a permutation of m integers randomly drawn from the range [0,n-1].
   * \param m Number of integers in permutation.
   * \param n Number of integers in the range.
   * \return The permutation.
   **/
  std::vector<int> random_permutation(int m, int n);
}

#include "oakStatistics.hpp"
