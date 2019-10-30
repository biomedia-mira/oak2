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

#include "miaImage.h"
#include "miaHistogram.h"

#include <Eigen/Dense>

namespace oak
{
  template<typename label_type>
  struct DataSample
  {
    int image_index;
    label_type label;
    Eigen::Vector3d point;
  };

  enum FeatureType
  {
    LOCAL_BOX,
    OFFSET_BOX,
    OFFSET_BOX_MINUS_LOCAL_POINT,
    OFFSET_BOX_MINUS_OFFSET_BOX,
    OFFSET_BOX_MINUS_LOCAL_POINT_BINARY,
    OFFSET_BOX_MINUS_OFFSET_BOX_BINARY,
    HISTOGRAM_MAX_BIN,
    HISTOGRAM_ENTROPY,
    OFFSET_POINT,
    OFFSET_POINT_MINUS_OFFSET_POINT,
  };

  struct Feature
  {
    FeatureType type;
    int channel_a;
    int channel_b;
    Eigen::Vector3d offset_a;
    Eigen::Vector3d offset_b;
    Eigen::Vector3d boxsize_a;
    Eigen::Vector3d boxsize_b;

    /**
     * \brief Serializes objects of type oak::Feature.
     **/
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
      ar & type;
      ar & channel_a;
      ar & channel_b;
      ar & offset_a;
      ar & offset_b;
      ar & boxsize_a;
      ar & boxsize_b;
    }
  };

  class FeatureFactory
  {
  public:
    FeatureFactory()
      : m_feature_types(0)
      , m_count_features_per_node(0)
      , m_count_thresholds(0)
      , m_min_offset(0)
      , m_max_offset(0)
      , m_min_boxsize(0)
      , m_max_boxsize(0)
	  , m_2d_features(false)
	  , m_cross_channel(false)
      , m_intensity_images()
      , m_integral_images()
    {
    }

    FeatureFactory(const std::vector<FeatureType> &featureTypes)
      : m_feature_types(featureTypes)
      , m_count_features_per_node(0)
      , m_count_thresholds(0)
      , m_min_offset(0)
      , m_max_offset(0)
      , m_min_boxsize(0)
      , m_max_boxsize(0)
	  , m_2d_features(false)
      , m_cross_channel(false)
      , m_intensity_images()
      , m_integral_images()
    {
    }

    /**
     * \brief Sets the number of features to be tested per node.
     * \param value The number of features.
     **/
    void count_features_per_node(int value)
    {
      m_count_features_per_node = value;
    }

    /**
     * \brief Gets the number of features to be tested per node.
     * \return The number of features.
     **/
    int count_features_per_node() const
    {
      return m_count_features_per_node;
    }

    /**
     * \brief Sets the number of thresholds to be tested per feature.
     * \param value The number of thresholds.
     **/
    void count_thresholds(int value)
    {
      m_count_thresholds = value;
    }

    /**
     * \brief Gets the number of thresholds to be tested per feature.
     * \return The number of thresholds.
     **/
    int count_thresholds() const
    {
      return m_count_thresholds;
    }

    /**
     * \brief Sets the parameters for randomized feature sampling.
     * \param countChannels The number of image channels.
     * \param minOffset The minimum feature offset (in mm).
     * \param maxOffset The maximum feature offset (in mm).
     * \param minBoxSize The minimum size of box features (in mm).
     * \param maxBoxSize The maximum size of box features (in mm).
     * \param crossChannel Enables cross channel features.
     **/
	void feature_parameters(int countChannels, double minOffset, double maxOffset, double minBoxSize, double maxBoxSize, bool crossChannel, bool use2dFeatures)
    {
      m_count_channels = countChannels;
      m_min_offset = minOffset;
      m_max_offset = maxOffset;
      m_min_boxsize = minBoxSize;
      m_max_boxsize = maxBoxSize;
	  m_2d_features = use2dFeatures;
	  m_cross_channel = crossChannel;
    }

    /**
     * \brief Sets the (multi-channel) intensity images.
     * \param intensityImages The intensity images.
     **/
    void intensity_images(const std::vector<std::vector<mia::Image>>& intensityImages)
    {
      m_intensity_images = intensityImages;
    }

    /**
     * \brief Sets the (multi-channel) integral images.
     * \param integralImages The integral images.
     **/
    void integral_images(const std::vector<std::vector<mia::Image>>& integralImages)
    {
      m_integral_images = integralImages;
    }

    /**
     * \brief Sets the (multi-channel) integral histograms.
     * \param integralHistograms The integral histograms.
     **/
    void integral_histograms(const std::vector<std::vector<mia::IntegralHistogram>>& integralHistograms)
    {
      m_integral_histograms = integralHistograms;
    }

    /**
     * \brief Samples a random feature.
     * \return The feature.
     **/
    Feature sample() const;

    /**
     * \brief Samples a set of random features.
     * \param numSamples Number of samples.
     * \return Feature set.
     **/
    std::vector<Feature> sample(int numSamples) const;

    /**
     * \brief Evaluates a feature for a given data sample.
     * \param feature The feature.
     * \param dataSample The data sample.
     * \return The feature response.
     **/
    template<typename label_type>
    float evaluate(const Feature& feature, const DataSample<label_type>& dataSample) const;

    /** \copydoc evaluate(const Feature&,const DataSample&) */
    template<typename label_type>
    float evaluate_local_box(const Feature& feature, const DataSample<label_type>& dataSample) const;
    /** \copydoc evaluate(const Feature&,const DataSample&) */
    template<typename label_type>
    float evaluate_offset_box(const Feature& feature, const DataSample<label_type>& dataSample) const;
    /** \copydoc evaluate(const Feature&,const DataSample&) */
    template<typename label_type>
    float evaluate_offset_box_minus_local_point(const Feature& feature, const DataSample<label_type>& dataSample) const;
    /** \copydoc evaluate(const Feature&,const DataSample&) */
    template<typename label_type>
    float evaluate_offset_box_minus_offset_box(const Feature& feature, const DataSample<label_type>& dataSample) const;
    /** \copydoc evaluate(const Feature&,const DataSample&) */
    template<typename label_type>
    float evaluate_offset_point(const Feature& feature, const DataSample<label_type>& dataSample) const;
    /** \copydoc evaluate(const Feature&,const DataSample&) */
    template<typename label_type>
    float evaluate_offset_point_minus_offset_point(const Feature& feature, const DataSample<label_type>& dataSample) const;

    /** \copydoc evaluate(const Feature&,const DataSample&) */
    template<typename label_type>
    float evaluate_offset_box_minus_local_point_binary(const Feature& feature, const DataSample<label_type>& dataSample) const;
    /** \copydoc evaluate(const Feature&,const DataSample&) */
    template<typename label_type>
    float evaluate_offset_box_minus_offset_box_binary(const Feature& feature, const DataSample<label_type>& dataSample) const;

    /** \copydoc evaluate(const Feature&,const DataSample&) */
    template<typename label_type>
    float evaluate_histogram_max_bin(const Feature& feature, const DataSample<label_type>& dataSample) const;
    /** \copydoc evaluate(const Feature&,const DataSample&) */
    template<typename label_type>
    float evaluate_histogram_entropy(const Feature& feature, const DataSample<label_type>& dataSample) const;

    /**
    * \brief Evaluates a point feature for a given data sample.
    * \param dataSample The data sample.
    * \param channel The image channel.
    * \param offset The offset.
    * \return The feature response.
    **/
    template<typename label_type>
    float evaluate_point_value(const DataSample<label_type>& dataSample, int channel, const Eigen::Vector3d& offset) const;

    /**
    * \brief Evaluates a box feature for a given data sample.
    * \param dataSample The data sample.
    * \param channel The image channel.
    * \param offset The offset.
    * \param boxsize The box size.
    * \return The feature response.
    **/
    template<typename label_type>
    float evaluate_box_value(const DataSample<label_type>& dataSample, int channel, const Eigen::Vector3d& offset, const Eigen::Vector3d& boxsize) const;

    /**
    * \brief Evaluates a box feature for a given data sample.
    * \param dataSample The data sample.
    * \param channel The image channel.
    * \param offset The offset.
    * \param boxsize The box size.
    * \return The feature response.
    **/
    template<typename label_type>
    mia::Histogram evaluate_box_histogram(const DataSample<label_type>& dataSample, int channel, const Eigen::Vector3d& offset, const Eigen::Vector3d& boxsize) const;

  private:
    std::vector<FeatureType> m_feature_types;
    int m_count_features_per_node;
    int m_count_thresholds;

    int m_count_channels;
    double m_min_offset;
    double m_max_offset;
    double m_min_boxsize;
    double m_max_boxsize;
	bool m_2d_features;
	bool m_cross_channel;
    bool m_binary_difference;

    std::vector<std::vector<mia::Image>> m_intensity_images;
    std::vector<std::vector<mia::Image>> m_integral_images;
    std::vector<std::vector<mia::IntegralHistogram>> m_integral_histograms;
  };
}

#include "oakFeatures.hpp"
