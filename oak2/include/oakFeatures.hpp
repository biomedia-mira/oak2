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

#include "miaImageProcessing.h"
#include <random>

namespace oak
{
  template<typename label_type>
  float FeatureFactory::evaluate(const Feature& feature, const DataSample<label_type>& dataSample) const
  {
    switch(feature.type)
    {
    case FeatureType::LOCAL_BOX:
      return evaluate_local_box(feature, dataSample);
    case FeatureType::OFFSET_BOX:
      return evaluate_offset_box(feature, dataSample);
    case FeatureType::OFFSET_BOX_MINUS_LOCAL_POINT:
      return evaluate_offset_box_minus_local_point(feature, dataSample);
    case FeatureType::OFFSET_BOX_MINUS_OFFSET_BOX:
      return evaluate_offset_box_minus_offset_box(feature, dataSample);
    case FeatureType::OFFSET_BOX_MINUS_LOCAL_POINT_BINARY:
      return evaluate_offset_box_minus_local_point_binary(feature, dataSample);
    case FeatureType::OFFSET_BOX_MINUS_OFFSET_BOX_BINARY:
      return evaluate_offset_box_minus_offset_box_binary(feature, dataSample);
    case FeatureType::HISTOGRAM_MAX_BIN:
      return evaluate_histogram_max_bin(feature, dataSample);
    case FeatureType::HISTOGRAM_ENTROPY:
      return evaluate_histogram_entropy(feature, dataSample);
    case FeatureType::OFFSET_POINT:
      return evaluate_offset_point(feature, dataSample);
    case FeatureType::OFFSET_POINT_MINUS_OFFSET_POINT:
      return evaluate_offset_point_minus_offset_point(feature, dataSample);
    default:
      return 0.0f;
    }
  }

  template<typename label_type>
  float FeatureFactory::evaluate_local_box(const Feature& feature, const DataSample<label_type>& dataSample) const
  {
    return evaluate_box_value(dataSample, feature.channel_a, Eigen::Vector3d(0.0, 0.0, 0.0), feature.boxsize_a);
  }

  template<typename label_type>
  float FeatureFactory::evaluate_offset_box(const Feature& feature, const DataSample<label_type>& dataSample) const
  {
    return evaluate_box_value(dataSample, feature.channel_a, feature.offset_a, feature.boxsize_a);
  }

  template<typename label_type>
  float FeatureFactory::evaluate_offset_box_minus_local_point(const Feature& feature, const DataSample<label_type>& dataSample) const
  {
    int img_index = dataSample.image_index;
    Eigen::Vector3d imgpoint = dataSample.point;
    float local_intensity = m_intensity_images[img_index][feature.channel_b](static_cast<int>(imgpoint[0]),static_cast<int>(imgpoint[1]),static_cast<int>(imgpoint[2]));

    float box_value = evaluate_box_value(dataSample, feature.channel_a, feature.offset_a, feature.boxsize_a);

    return box_value - local_intensity;
  }

  template<typename label_type>
  float FeatureFactory::evaluate_offset_box_minus_offset_box(const Feature& feature, const DataSample<label_type>& dataSample) const
  {
    float box_value_a = evaluate_box_value(dataSample, feature.channel_a, feature.offset_a, feature.boxsize_a);
    float box_value_b = evaluate_box_value(dataSample, feature.channel_b, feature.offset_b, feature.boxsize_b);

    return box_value_a - box_value_b;
  }

  template<typename label_type>
  float FeatureFactory::evaluate_offset_point(const Feature& feature, const DataSample<label_type>& dataSample) const
  {
    return evaluate_point_value(dataSample, feature.channel_a, feature.offset_a);
  }

  template<typename label_type>
  float FeatureFactory::evaluate_offset_point_minus_offset_point(const Feature& feature, const DataSample<label_type>& dataSample) const
  {
    return evaluate_point_value(dataSample, feature.channel_a, feature.offset_a) - evaluate_point_value(dataSample, feature.channel_b, feature.offset_b);
  }

  template<typename label_type>
  float FeatureFactory::evaluate_offset_box_minus_local_point_binary(const Feature& feature, const DataSample<label_type>& dataSample) const
  {
    int img_index = dataSample.image_index;
    Eigen::Vector3d imgpoint = dataSample.point;
    float local_intensity = m_intensity_images[img_index][feature.channel_b](static_cast<int>(imgpoint[0]), static_cast<int>(imgpoint[1]), static_cast<int>(imgpoint[2]));

    float box_value = evaluate_box_value(dataSample, feature.channel_a, feature.offset_a, feature.boxsize_a);

    return box_value > local_intensity ? 1.0f : -1.0f;
  }

  template<typename label_type>
  float FeatureFactory::evaluate_offset_box_minus_offset_box_binary(const Feature& feature, const DataSample<label_type>& dataSample) const
  {
    float box_value_a = evaluate_box_value(dataSample, feature.channel_a, feature.offset_a, feature.boxsize_a);
    float box_value_b = evaluate_box_value(dataSample, feature.channel_b, feature.offset_b, feature.boxsize_b);

    return box_value_b > box_value_a ? 1.0f : -1.0f;
  }

  template<typename label_type>
  float FeatureFactory::evaluate_histogram_max_bin(const Feature& feature, const DataSample<label_type>& dataSample) const
  {
    auto histogram = evaluate_box_histogram(dataSample, feature.channel_a, feature.offset_a, feature.boxsize_a);

    if (histogram.totalCount() > 0)
      return static_cast<float>(mia::maxCountIndex(histogram.counts()));
    else
      return 0.0f;
  }

  template<typename label_type>
  float FeatureFactory::evaluate_histogram_entropy(const Feature& feature, const DataSample<label_type>& dataSample) const
  {
    auto histogram = evaluate_box_histogram(dataSample, feature.channel_a, feature.offset_a, feature.boxsize_a);

    if (histogram.totalCount() > 0)
      return static_cast<float>(mia::entropy(histogram.counts(), histogram.totalCount()));
    else
      return 0.0f;
  }

  template<typename label_type>
  float FeatureFactory::evaluate_point_value(const DataSample<label_type>& dataSample, int channel, const Eigen::Vector3d& offset) const
  {
    int img_index = dataSample.image_index;
    Eigen::Vector3d spacing = m_intensity_images[img_index][channel].spacing();
    Eigen::Vector3d imgpoint = dataSample.point + offset.cwiseQuotient(spacing);

    int sizeX = m_intensity_images[img_index][channel].sizeX();
    int sizeY = m_intensity_images[img_index][channel].sizeY();
    int sizeZ = m_intensity_images[img_index][channel].sizeZ();

    if (imgpoint[0] < -0.5 || imgpoint[1] < -0.5 || imgpoint[2] < -0.5 || imgpoint[0] >= static_cast<double>(sizeX)-0.5 || imgpoint[1] >= static_cast<double>(sizeY)-0.5 || imgpoint[2] >= static_cast<double>(sizeZ)-0.5)
    {
      return 0.0f;
    }

    int x = static_cast<int>(imgpoint[0] + 0.5);
    int y = static_cast<int>(imgpoint[1] + 0.5);
    int z = static_cast<int>(imgpoint[2] + 0.5);

    return m_intensity_images[img_index][channel](x, y, z);
  }

  template<typename label_type>
  float FeatureFactory::evaluate_box_value(const DataSample<label_type>& dataSample, int channel, const Eigen::Vector3d& offset, const Eigen::Vector3d& boxsize) const
  {
    int img_index = dataSample.image_index;
    Eigen::Vector3d imgpoint = dataSample.point;
    Eigen::Vector3d spacing = m_intensity_images[img_index][channel].spacing();
    Eigen::Vector3d boxsize_pixels = boxsize.cwiseQuotient(spacing);
    Eigen::Vector3d boxpoint_min = imgpoint + offset.cwiseQuotient(spacing) - boxsize_pixels / 2.0;
    Eigen::Vector3d boxpoint_max = boxpoint_min + boxsize_pixels;

    int x_min = static_cast<int>(boxpoint_min[0] + 0.5);
    int y_min = static_cast<int>(boxpoint_min[1] + 0.5);
    int z_min = static_cast<int>(boxpoint_min[2] + 0.5);

    int x_max = static_cast<int>(boxpoint_max[0] + 0.5);
    int y_max = static_cast<int>(boxpoint_max[1] + 0.5);
    int z_max = static_cast<int>(boxpoint_max[2] + 0.5);

    int sizeX = m_integral_images[img_index][channel].sizeX();
    int sizeY = m_integral_images[img_index][channel].sizeY();
    int sizeZ = m_integral_images[img_index][channel].sizeZ();

    int count_outside_x = 0;
    int count_outside_y = 0;
    int count_outside_z = 0;

    if (boxpoint_min[0] < -0.5)
    {
      x_min = 0;
      count_outside_x++;
    }
    else if (boxpoint_min[0] >= static_cast<double>(sizeX)-0.5)
    {
      x_min = sizeX-1;
      count_outside_x++;
    }

    if (boxpoint_max[0] < -0.5)
    {
      x_max = 0;
      count_outside_x++;
    }
    else if (boxpoint_max[0] >= static_cast<double>(sizeX)-0.5)
    {
      x_max = sizeX-1;
      count_outside_x++;
    }

    if (boxpoint_min[1] < -0.5)
    {
      y_min = 0;
      count_outside_y++;
    }
    else if (boxpoint_min[1] >= static_cast<double>(sizeY)-0.5)
    {
      y_min = sizeY-1;
      count_outside_y++;
    }

    if (boxpoint_max[1] < -0.5)
    {
      y_max = 0;
      count_outside_y++;
    }
    else if (boxpoint_max[1] >= static_cast<double>(sizeY)-0.5)
    {
      y_max = sizeY-1;
      count_outside_y++;
    }

    if (boxpoint_min[2] < -0.5)
    {
      z_min = 0;
      count_outside_z++;
    }
    else if (boxpoint_min[2] >= static_cast<double>(sizeZ)-0.5)
    {
      z_min = sizeZ-1;
      count_outside_z++;
    }

    if (boxpoint_max[2] < -0.5)
    {
      z_max = 0;
      count_outside_z++;
    }
    else if (boxpoint_max[2] >= static_cast<double>(sizeZ)-0.5)
    {
      z_max = sizeZ-1;
      count_outside_z++;
    }

    if (count_outside_x == 2 || count_outside_y == 2 || count_outside_z == 2) return 0.0f;

    float integral_value = evaluate_integral_image(m_integral_images[img_index][channel], x_min, y_min, z_min, x_max, y_max, z_max);

    float boxvolume = static_cast<float>((x_max - x_min) * (y_max - y_min) * (z_max - z_min));
    if (boxvolume == 0.0f) boxvolume = 1.0;

    return integral_value / static_cast<float>(boxvolume);
  }

  template<typename label_type>
  mia::Histogram FeatureFactory::evaluate_box_histogram(const DataSample<label_type>& dataSample, int channel, const Eigen::Vector3d& offset, const Eigen::Vector3d& boxsize) const
  {
    int img_index = dataSample.image_index;
    Eigen::Vector3d imgpoint = dataSample.point;
    Eigen::Vector3d spacing = m_intensity_images[img_index][channel].spacing();
    Eigen::Vector3d boxsize_pixels = boxsize.cwiseQuotient(spacing);
    Eigen::Vector3d boxpoint_min = imgpoint + offset.cwiseQuotient(spacing) - boxsize_pixels / 2.0;
    Eigen::Vector3d boxpoint_max = boxpoint_min + boxsize_pixels;

    int x_min = static_cast<int>(boxpoint_min[0] + 0.5);
    int y_min = static_cast<int>(boxpoint_min[1] + 0.5);
    int z_min = static_cast<int>(boxpoint_min[2] + 0.5);

    int x_max = static_cast<int>(boxpoint_max[0] + 0.5);
    int y_max = static_cast<int>(boxpoint_max[1] + 0.5);
    int z_max = static_cast<int>(boxpoint_max[2] + 0.5);

    int sizeX = m_integral_images[img_index][channel].sizeX();
    int sizeY = m_integral_images[img_index][channel].sizeY();
    int sizeZ = m_integral_images[img_index][channel].sizeZ();

    int count_outside_x = 0;
    int count_outside_y = 0;
    int count_outside_z = 0;

    if (boxpoint_min[0] < -0.5)
    {
      x_min = 0;
      count_outside_x++;
    }
    else if (boxpoint_min[0] >= static_cast<double>(sizeX)-0.5)
    {
      x_min = sizeX - 1;
      count_outside_x++;
    }

    if (boxpoint_max[0] < -0.5)
    {
      x_max = 0;
      count_outside_x++;
    }
    else if (boxpoint_max[0] >= static_cast<double>(sizeX)-0.5)
    {
      x_max = sizeX - 1;
      count_outside_x++;
    }

    if (boxpoint_min[1] < -0.5)
    {
      y_min = 0;
      count_outside_y++;
    }
    else if (boxpoint_min[1] >= static_cast<double>(sizeY)-0.5)
    {
      y_min = sizeY - 1;
      count_outside_y++;
    }

    if (boxpoint_max[1] < -0.5)
    {
      y_max = 0;
      count_outside_y++;
    }
    else if (boxpoint_max[1] >= static_cast<double>(sizeY)-0.5)
    {
      y_max = sizeY - 1;
      count_outside_y++;
    }

    if (boxpoint_min[2] < -0.5)
    {
      z_min = 0;
      count_outside_z++;
    }
    else if (boxpoint_min[2] >= static_cast<double>(sizeZ)-0.5)
    {
      z_min = sizeZ - 1;
      count_outside_z++;
    }

    if (boxpoint_max[2] < -0.5)
    {
      z_max = 0;
      count_outside_z++;
    }
    else if (boxpoint_max[2] >= static_cast<double>(sizeZ)-0.5)
    {
      z_max = sizeZ - 1;
      count_outside_z++;
    }

    if (count_outside_x == 2 || count_outside_y == 2 || count_outside_z == 2) return mia::Histogram();

    return evaluate_integral_histogram(m_integral_histograms[img_index][channel], x_min, y_min, z_min, x_max, y_max, z_max);
  }
}
