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

#include "oakFeatures.h"

namespace oak
{
  Feature FeatureFactory::sample() const
  {
    std::random_device rd;
    std::mt19937 mt(rd());

    std::uniform_int_distribution<int> typ(0, static_cast<int>(m_feature_types.size()) - 1);
    std::uniform_int_distribution<int> cha(0, m_count_channels - 1);
    std::uniform_real_distribution<double> off(m_min_offset, m_max_offset);
    std::uniform_real_distribution<double> box(m_min_boxsize, m_max_boxsize);
    std::uniform_int_distribution<int> sign(0, 1);

    int type = typ(mt);
    int channel_a = cha(mt);
    int channel_b = m_cross_channel ? cha(mt) : channel_a;

    //OFFSET SAMPLING FROM CUBOID
    double s = sign(mt) == 1 ? 1.0 : -1.0;
    double offset_a_x = off(mt) * s;
    s = sign(mt) == 1 ? 1.0 : -1.0;
    double offset_a_y = off(mt) * s;
    s = sign(mt) == 1 ? 1.0 : -1.0;
    double offset_a_z = off(mt) * s;
    s = sign(mt) == 1 ? 1.0 : -1.0;
    double offset_b_x = off(mt) * s;
    s = sign(mt) == 1 ? 1.0 : -1.0;
    double offset_b_y = off(mt) * s;
    s = sign(mt) == 1 ? 1.0 : -1.0;
    double offset_b_z = off(mt) * s;

    // OFFSET SAMPLING FROM SPHERE (NEEDS MODIFICATION TO INCLUDE MIN_OFFSET CONSTRAINT)
    //std::normal_distribution<double> offn(0.0, 1.0);
    //std::uniform_real_distribution<double> offd(0.0, 1.0);
    //
    //double dx1 = offn(mt);
    //double dy1 = offn(mt);
    //double dz1 = offn(mt);
    //double n1 = sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1);
    //double d1 = pow(offd(mt), 1.0 / 3.0);
    //double offset_a_x = 0.0, offset_a_y = 0.0, offset_a_z = 0.0;
    //if (n1 > 0.0)
    //{
    //  offset_a_x = dx1 / n1 * d1 * m_max_offset;
    //  offset_a_y = dy1 / n1 * d1 * m_max_offset;
    //  offset_a_z = dz1 / n1 * d1 * m_max_offset;
    //}

    //double dx2 = offn(mt);
    //double dy2 = offn(mt);
    //double dz2 = offn(mt);
    //double n2 = sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2);
    //double d2 = pow(offd(mt), 1.0 / 3.0);
    //double offset_b_x = 0.0, offset_b_y = 0.0, offset_b_z = 0.0;
    //if (n2 > 0.0)
    //{
    //  offset_b_x = dx2 / n2 * d2 * m_max_offset;
    //  offset_b_y = dy2 / n2 * d2 * m_max_offset;
    //  offset_b_z = dz2 / n2 * d2 * m_max_offset;
    //}

    double boxsize_a_x = box(mt);
    double boxsize_a_y = box(mt);
    double boxsize_a_z = box(mt);
    double boxsize_b_x = box(mt);
    double boxsize_b_y = box(mt);
    double boxsize_b_z = box(mt);

	if (m_2d_features)
	{
		offset_a_z = 0.0;
		offset_b_z = 0.0;
		boxsize_a_z = 1.0;
		boxsize_b_z = 1.0;
	}

	Feature feature;
    feature.type = m_feature_types[type];
    feature.channel_a = channel_a;
    feature.channel_b = channel_b;
    feature.offset_a = Eigen::Vector3d(offset_a_x, offset_a_y, offset_a_z);
    feature.offset_b = Eigen::Vector3d(offset_b_x, offset_b_y, offset_b_z);
    feature.boxsize_a = Eigen::Vector3d(boxsize_a_x, boxsize_a_y, boxsize_a_z);
    feature.boxsize_b = Eigen::Vector3d(boxsize_b_x, boxsize_b_y, boxsize_b_z);

    return feature;
  }

  std::vector<Feature> FeatureFactory::sample(int numSamples) const
  {
    std::vector<Feature> feature_set;
    for (int i = 0; i < numSamples; i++)
    {
      feature_set.push_back(sample());
    }

    return feature_set;
  }
}
