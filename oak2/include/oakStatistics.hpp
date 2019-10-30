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

#include <utility>
#include <random>

namespace oak
{
  template<typename label_type>
  float evaluate_splitfunction(const SplitFunction& function, const DataSample<label_type>& sample, const FeatureFactory& factory, const std::vector<Feature>& featureSet)
  {
    float feature_response = factory.evaluate(featureSet[function.feature_index], sample);

    return (function.threshold - feature_response);
  }
}
