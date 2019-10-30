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

#include "oakStatistics.h"

#include <string>
#include <fstream>
#include <memory>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

namespace oak
{
  template<typename predictor_type>
  struct Node
  {
    bool is_leaf;
    int depth;
    double split_entropy;
    SplitFunction split_function;
    predictor_type predictor;
    boost::shared_ptr<Node> left;
    boost::shared_ptr<Node> right;

    /**
     * \brief Serializes objects of type oak::Node.
     **/
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
      ar & is_leaf;
      ar & depth;
      ar & split_entropy;
      ar & split_function;
      ar & predictor;
      ar & left;
      ar & right;
    }
  };

  /**
   * \brief Trains a tree from the root node.
   * \param root The root node of the tree to be trained.
   * \param dataSamples The set of data samples.
   * \param featureFactory The feature factory instance.
   * \param featureSet The feature set.
   * \param countLabels The number of categorical labels.
   * \param maxDepth The maximum tree depth.
   * \param minSamples The minimum number of samples allowed in a split.
   **/
  void train_tree(Node<Classifier>& root, const std::vector<DataSample<int>*>& dataSamples, const FeatureFactory& featureFactory, const std::vector<Feature>& featureSet, int countLabels, int maxDepth, int minSamples, bool improveParent);

  /**
   * \brief Trains a node by determining the optimal split function.
   * \param node The node to be trained.
   * \param dataSamples The set of data samples.
   * \param featureFactory The feature factory instance.
   * \param featureSet The feature set.
   * \param countLabels The number of categorical labels.
   * \param minSamples The minimum number of samples allowed in a split.
   * \param labelWeights The label weights used for rebalancing the contribution of data samples.
   **/
  void train_node(Node<Classifier>& node, const std::vector<DataSample<int>*>& dataSamples, const FeatureFactory& featureFactory, const std::vector<Feature>& featureSet, int countLabels, int minSamples, bool improveParent, const Eigen::ArrayXd& labelWeights);

  /**
   * \brief Determines split candidates for a set of data samples and a given feature.
   * \param dataSamples The set of data samples.
   * \param featureIndex The feature index.
   * \param featureFactory The feature factory instance.
   * \param featureSet The feature set.
   * \param countLabels The number of categorical labels.
   * \param minSamples The minimum number of samples allowed in a split.
   * \param labelWeights The label weights used for rebalancing the contribution of data samples.
   * \param entropyParent The entropy of the label distribution of the parent node.
   * \return A split candidate.
   **/
  SplitCandidate<Classifier> split_candidate(const std::vector<DataSample<int>*>& dataSamples, int featureIndex, const FeatureFactory& featureFactory, const std::vector<Feature>& featureSet, int countLabels, int minSamples, const Eigen::ArrayXd& labelWeights, double entropyParent);

  /**
   * \brief Trains a tree from the root node.
   * \param root The root node of the tree to be trained.
   * \param dataSamples The set of data samples.
   * \param featureFactory The feature factory instance.
   * \param featureSet The feature set.
   * \param countVariables The number of regression variables.
   * \param maxDepth The maximum tree depth.
   * \param minSamples The minimum number of samples allowed in a split.
   **/
  void train_tree(Node<Regressor>& root, const std::vector<DataSample<Eigen::VectorXd>*>& dataSamples, const FeatureFactory& featureFactory, const std::vector<Feature>& featureSet, int countVariables, int maxDepth, int minSamples, bool improveParent);

  /**
   * \brief Trains a node by determining the optimal split function.
   * \param node The node to be trained.
   * \param dataSamples The set of data samples.
   * \param featureFactory The feature factory instance.
   * \param featureSet The feature set.
   * \param countVariables The number of regression variables.
   * \param minSamples The minimum number of samples allowed in a split.
   **/
  void train_node(Node<Regressor>& node, const std::vector<DataSample<Eigen::VectorXd>*>& dataSamples, const FeatureFactory& featureFactory, const std::vector<Feature>& featureSet, int countVariables, int minSamples, bool improveParent);

  /**
   * \brief Determines split candidates for a set of data samples and a given feature.
   * \param dataSamples The set of data samples.
   * \param featureIndex The feature index.
   * \param featureFactory The feature factory instance.
   * \param featureSet The feature set.
   * \param countVariables The number of regression variables.
   * \param minSamples The minimum number of samples allowed in a split.
   * \param entropyParent The entropy of the label distribution of the parent node.
   * \return A split candidate.
   **/
  SplitCandidate<Regressor> split_candidate(const std::vector<DataSample<Eigen::VectorXd>*>& dataSamples, int featureIndex, const FeatureFactory& featureFactory, const std::vector<Feature>& featureSet, int countVariables, int minSamples, double entropyParent);

  /**
   * \brief Splits a set of data samples into two subsets.
   * \param dataSamples The set of data samples.
   * \param function The split function.
   * \param featureFactory The feature factory instance.
   * \param featureSet The feature set.
   * \return Two subsets of data samples.
   **/
  template<typename label_type>
  std::pair<std::vector<DataSample<label_type>*>, std::vector<DataSample<label_type>*>> split_samples(const std::vector<DataSample<label_type>*>& dataSamples, const SplitFunction& function, const FeatureFactory& featureFactory, const std::vector<Feature>& featureSet);

  /**
   * \brief Descends a tree for a given data sample down to a maximum depth or leaf node.
   * \param root The root node of the tree.
   * \param samples The data sample used for updating statistics.
   * \param featureFactory The feature factory instance.
   * \param featureSet The feature set.
   * \param maxDepth The maximum depth.
   * \return The reached node.
   **/
  template<typename label_type, typename predictor_type>
  const Node<predictor_type>& descend_tree(const Node<predictor_type>& root, const DataSample<label_type>& sample, const FeatureFactory& featureFactory, const std::vector<Feature>& featureSet, int maxDepth);

  /**
  * \brief Descends a tree for a given data sample down to a maximum depth or leaf node and returns the node index.
  * \param root The root node of the tree.
  * \param samples The data sample used for updating statistics.
  * \param featureFactory The feature factory instance.
  * \param featureSet The feature set.
  * \param maxDepth The maximum depth.
  * \return The reached node index.
  **/
  template<typename label_type, typename predictor_type>
  int node_index(const Node<predictor_type>& root, const DataSample<label_type>& sample, const FeatureFactory& featureFactory, const std::vector<Feature>& featureSet, int maxDepth);

  /**
  * \brief Descends a tree for a given data sample down to a maximum depth or leaf node and returns both the node and its index.
  * \param root The root node of the tree.
  * \param samples The data sample used for updating statistics.
  * \param featureFactory The feature factory instance.
  * \param featureSet The feature set.
  * \param maxDepth The maximum depth.
  * \return The reached node and its index.
  **/
  template<typename label_type, typename predictor_type>
  std::pair<const Node<predictor_type>&, int> descend_tree_node_index(const Node<predictor_type>& root, const DataSample<label_type>& sample, const FeatureFactory& featureFactory, const std::vector<Feature>& featureSet, int maxDepth);

  /**
  * \brief Descends a tree for a given data sample down to a maximum depth or leaf node and returns both the path of nodes and the leaf index.
  * \param root The root node of the tree.
  * \param samples The data sample used for updating statistics.
  * \param featureFactory The feature factory instance.
  * \param featureSet The feature set.
  * \param maxDepth The maximum depth.
  * \return The path of nodes and leaf index.
  **/
  template<typename label_type, typename predictor_type>
  std::pair<std::vector<const Node<predictor_type>*>, int> descend_tree_path_node_index(const Node<predictor_type>& root, const DataSample<label_type>& sample, const FeatureFactory& featureFactory, const std::vector<Feature>& featureSet, int maxDepth);

  /**
   * \brief Updates the data statistics stored in the predictors of tree nodes.
   * \param root The root node of the tree.
   * \param dataSamples The data samples used for updating statistics.
   * \param featureFactory The feature factory instance.
   * \param featureSet The feature set.
   **/
  template<typename label_type, typename predictor_type>
  void update_tree(Node<predictor_type>& root, const std::vector<DataSample<label_type>*>& dataSamples, const FeatureFactory& featureFactory, const std::vector<Feature>& featureSet);

  /**
   * \brief Saves a serializable object under the given filename.
   * \param serializable_type The object which is saved.
   * \param filename The filename under which the object is saved.
   **/
  template <typename SerializableType>
  void save( const SerializableType& serializable_type, const std::string& filename )
  {
    std::ofstream ofs(filename, std::ios::binary);
    boost::archive::binary_oarchive oa( ofs );
    oa << serializable_type;
  }

  /** \copydoc save(SerializableType&,const std::string&) */
  template <typename SerializableType>
  void save_text( const SerializableType& serializable_type, const std::string& filename )
  {
    std::ofstream ofs(filename);
    boost::archive::text_oarchive oa( ofs );
    oa << serializable_type;
  }

  /**
   * \brief Loads a serializable object with the given filename.
   * \param serializable_type The object in which the loaded data is stored.
   * \param filename The filename of the object which is loaded.
   **/
  template <typename SerializableType>
  void load( SerializableType& serializable_type, const std::string& filename )
  {
    std::ifstream ifs(filename, std::ios::binary);
    boost::archive::binary_iarchive ia( ifs );
    ia >> serializable_type;
  }

  /** \copydoc load(SerializableType&,const std::string&) */
  template <typename SerializableType>
  void load_text( SerializableType& serializable_type, const std::string& filename )
  {
    std::ifstream ifs(filename);
    boost::archive::text_iarchive ia( ifs );
    ia >> serializable_type;
  }
}

namespace boost
{
namespace serialization
{
  /**
   * \brief Serializes objects of type Eigen::Matrix.
   **/
  template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
  inline void serialize(Archive & ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> & t, const unsigned int file_version)
  {
    size_t rows = t.rows(), cols = t.cols();
    ar & rows;
    ar & cols;
    if( rows * cols != t.size() )
      t.resize( rows, cols );

    ar & boost::serialization::make_array(t.data(), t.size());
  }

  /**
   * \brief Serializes objects of type Eigen::Array.
   **/
  template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
  inline void serialize(Archive & ar, Eigen::Array<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> & t, const unsigned int file_version)
  {
    size_t rows = t.rows(), cols = t.cols();
    ar & rows;
    ar & cols;
    if( rows * cols != t.size() )
      t.resize( rows, cols );

    ar & boost::serialization::make_array(t.data(), t.size());
  }
}
}
#include "oakForest.hpp"
