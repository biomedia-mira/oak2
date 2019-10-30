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

#define _WIN32_WINNT 0x0601

#include "itkio.h"
#include "oakForest.h"
#include "miaImageProcessing.h"

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <chrono>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

using namespace mia;
using namespace oak;

namespace po = boost::program_options;
namespace fs = boost::filesystem;

template <typename T>
void strings_to_values(const std::vector<std::string>& string_seq, std::vector<T>& values)
{
  for(std::vector<std::string>::const_iterator it = string_seq.begin(); it != string_seq.end(); ++it)
  {
    std::stringstream ss(*it);
    std::copy(std::istream_iterator<T>(ss), std::istream_iterator<T>(), back_inserter(values));
  }
}

void params_to_vector(const std::vector<double>& params, Eigen::Vector3d& vector)
{
  if (params.size() > 0)
  {
    vector[0] = (params.size() > 0) ? params[0] : 0;
    vector[1] = (params.size() > 1) ? params[1] : params[0];
    vector[2] = (params.size() > 2) ? params[2] : params[0];
  }
  else
  {
    vector.setZero();
  }
}

void update_feature_map_with_point_feature(const Eigen::Vector3d &sample_point, const Eigen::Vector3d& offset, Image &feature_map)
{
  Eigen::Vector3d spacing = feature_map.spacing();

  int sizeX = feature_map.sizeX();
  int sizeY = feature_map.sizeY();
  int sizeZ = feature_map.sizeZ();

  Eigen::Vector3d imgpoint = sample_point + offset.cwiseQuotient(spacing);

  if (!(imgpoint[0] < -0.5 || imgpoint[1] < -0.5 || imgpoint[2] < -0.5 || imgpoint[0] >= static_cast<double>(sizeX)-0.5 || imgpoint[1] >= static_cast<double>(sizeY)-0.5 || imgpoint[2] >= static_cast<double>(sizeZ)-0.5))
  {
    int x = static_cast<int>(imgpoint[0] + 0.5);
    int y = static_cast<int>(imgpoint[1] + 0.5);
    int z = static_cast<int>(imgpoint[2] + 0.5);

    feature_map(x, y, z) = feature_map(x, y, z) + 1;
  }
}

void update_feature_map_with_box_feature(const Eigen::Vector3d &sample_point, const Eigen::Vector3d& offset, const Eigen::Vector3d& boxsize, Image &feature_map)
{
  Eigen::Vector3d spacing = feature_map.spacing();

  int sizeX = feature_map.sizeX();
  int sizeY = feature_map.sizeY();
  int sizeZ = feature_map.sizeZ();

  Eigen::Vector3d boxsize_pixels = boxsize.cwiseQuotient(spacing);
  Eigen::Vector3d boxpoint_min = sample_point + offset.cwiseQuotient(spacing) - boxsize_pixels / 2.0;
  Eigen::Vector3d boxpoint_max = boxpoint_min + boxsize_pixels;

  int x_min = static_cast<int>(boxpoint_min[0] + 0.5);
  int y_min = static_cast<int>(boxpoint_min[1] + 0.5);
  int z_min = static_cast<int>(boxpoint_min[2] + 0.5);

  int x_max = static_cast<int>(boxpoint_max[0] + 0.5);
  int y_max = static_cast<int>(boxpoint_max[1] + 0.5);
  int z_max = static_cast<int>(boxpoint_max[2] + 0.5);

  int count_outside_x = 0;
  int count_outside_y = 0;
  int count_outside_z = 0;

  if (boxpoint_min[0] < -0.5)
  {
    x_min = 0;
    count_outside_x++;
  }
  else if (boxpoint_min[0] >= static_cast<double>(sizeX + 1) - 0.5)
  {
    x_min = sizeX;
    count_outside_x++;
  }

  if (boxpoint_max[0] < -0.5)
  {
    x_max = 0;
    count_outside_x++;
  }
  else if (boxpoint_max[0] >= static_cast<double>(sizeX + 1) - 0.5)
  {
    x_max = sizeX;
    count_outside_x++;
  }

  if (boxpoint_min[1] < -0.5)
  {
    y_min = 0;
    count_outside_y++;
  }
  else if (boxpoint_min[1] >= static_cast<double>(sizeY + 1) - 0.5)
  {
    y_min = sizeY;
    count_outside_y++;
  }

  if (boxpoint_max[1] < -0.5)
  {
    y_max = 0;
    count_outside_y++;
  }
  else if (boxpoint_max[1] >= static_cast<double>(sizeY + 1) - 0.5)
  {
    y_max = sizeY;
    count_outside_y++;
  }

  if (boxpoint_min[2] < -0.5)
  {
    z_min = 0;
    count_outside_z++;
  }
  else if (boxpoint_min[2] >= static_cast<double>(sizeZ + 1) - 0.5)
  {
    z_min = sizeZ;
    count_outside_z++;
  }

  if (boxpoint_max[2] < -0.5)
  {
    z_max = 0;
    count_outside_z++;
  }
  else if (boxpoint_max[2] >= static_cast<double>(sizeZ + 1) - 0.5)
  {
    z_max = sizeZ;
    count_outside_z++;
  }

  if (!(count_outside_x == 2 || count_outside_y == 2 || count_outside_z == 2))
  {
    for (int z = z_min; z < z_max; z++)
    {
      for (int y = y_min; y < y_max; y++)
      {
        for (int x = x_min; x < x_max; x++)
        {
          feature_map(x, y, z) = feature_map(x, y, z) + 1;
        }
      }
    }
  }
}

void update_feature_map(const Feature &feature, const Eigen::Vector3d &sample_point, Image &feature_map)
{
  if (feature.type == oak::OFFSET_POINT || feature.type == oak::OFFSET_POINT_MINUS_OFFSET_POINT)
  {
    update_feature_map_with_point_feature(sample_point, feature.offset_a, feature_map);
  }
  else if (feature.type == oak::LOCAL_BOX || feature.type == oak::OFFSET_BOX || feature.type == oak::OFFSET_BOX_MINUS_LOCAL_POINT || feature.type == oak::OFFSET_BOX_MINUS_OFFSET_BOX || feature.type == oak::OFFSET_BOX_MINUS_LOCAL_POINT_BINARY || feature.type == oak::OFFSET_BOX_MINUS_OFFSET_BOX_BINARY)
  {
    update_feature_map_with_box_feature(sample_point, feature.offset_a, feature.boxsize_a, feature_map);
  }

  if (feature.type == oak::OFFSET_POINT_MINUS_OFFSET_POINT)
  {
    update_feature_map_with_point_feature(sample_point, feature.offset_b, feature_map);
  }
  else if (feature.type == oak::OFFSET_BOX_MINUS_OFFSET_BOX || feature.type == oak::OFFSET_BOX_MINUS_OFFSET_BOX_BINARY)
  {
    update_feature_map_with_box_feature(sample_point, feature.offset_b, feature.boxsize_b, feature_map);
  }
}

int main(int argc, char *argv[])
{
  std::string config_file;

  std::vector<std::string> simagelists;
  std::vector<std::string> smin_hist_values;
  std::vector<std::string> smax_hist_values;
  std::vector<std::string> sspacing;
  std::vector<std::string> ssmoothing;
  std::string labelfile;
  std::string masklist;
  std::string output_path;

  bool write_temp = false;

  int count_trees = 0;
  int max_depth = 0;
  int min_samples = 0;
  int count_features_total = 0;
  int count_features_per_node = 0;
  int count_thresholds = 0;
  bool use_point_intensity_features = true;
  bool use_abs_intensity_features = false;
  bool use_diff_intensity_features = false;
  bool use_histogram_features = false;
  int histogram_bins = 0;
  double min_offset = 0;
  double max_offset = 0;
  double min_boxsize = 0;
  double max_boxsize = 0;
  bool use_2d_features = false;
  bool use_cross_channel_features = false;
  bool use_binary_features = false;
  float data_sampling_rate = 0;
  bool strip_images = false;
  bool reorient_images = false;
  bool improve_parent = true;

  try
  {
    // Declare the supported options.
    po::options_description generic("generic options");
    generic.add_options()
    ("help", "produce help message")
    ("config", po::value<std::string>(&config_file), "configuration file")
    ("temp", po::bool_switch(&write_temp)->default_value(false), "enable output of temporary files")
    ;

    po::options_description config("specific options");
    config.add_options()
    ("output", po::value<std::string>(&output_path), "output path")
    ("images", po::value<std::vector<std::string>>(&simagelists)->multitoken(), "text file(s) listing images")
    ("labels", po::value<std::string>(&labelfile), "text file containing the target labels")
    ("masks", po::value<std::string>(&masklist), "text file listing sampling masks")
    ("trees", po::value<int>(&count_trees), "number of trees")
    ("depth", po::value<int>(&max_depth), "maxmimum tree depth")
    ("min_samples", po::value<int>(&min_samples), "minimum number of samples in leaf nodes")
    ("features_total", po::value<int>(&count_features_total), "number of total random features")
    ("features_per_node", po::value<int>(&count_features_per_node), "number of random features tested per node")
    ("thresholds", po::value<int>(&count_thresholds), "number of thresholds per feature")
    ("point_intensity_features", po::bool_switch(&use_point_intensity_features)->default_value(true), "enable point intensity features")
    ("abs_intensity_features", po::bool_switch(&use_abs_intensity_features)->default_value(false), "enable absolute intensity features")
    ("diff_intensity_features", po::bool_switch(&use_diff_intensity_features)->default_value(false), "enable intensity difference features")
    ("histogram_features", po::bool_switch(&use_histogram_features)->default_value(false), "enable histogram features")
    ("histogram_bins", po::value<int>(&histogram_bins), "number of bins for histogram features")
    ("histogram_min_values", po::value<std::vector<std::string>>(&smin_hist_values)->multitoken(), "minimum intensity value(s) for histogram features")
    ("histogram_max_values", po::value<std::vector<std::string>>(&smax_hist_values)->multitoken(), "maximum intensity value(s) for histogram features")
    ("min_offset", po::value<double>(&min_offset), "minimum feature offset (in mm)")
    ("max_offset", po::value<double>(&max_offset), "maximum feature offset (in mm)")    
    ("min_boxsize", po::value<double>(&min_boxsize), "minimum feature box size (in mm)")
	("max_boxsize", po::value<double>(&max_boxsize), "maxmimum feature box size (in mm)")
	("use_2d_features", po::bool_switch(&use_2d_features)->default_value(false), "enable 2D features")
    ("cross_channel", po::bool_switch(&use_cross_channel_features)->default_value(false), "enable cross channel features")
    ("binary_features", po::bool_switch(&use_binary_features)->default_value(false), "enable binary features")
    ("spacing", po::value<std::vector<std::string>>(&sspacing)->multitoken(), "element spacing for image resampling (in mm)")
    ("smoothing", po::value<std::vector<std::string>>(&ssmoothing)->multitoken(), "sigmas for Gaussian image smoothing (in mm)")
    ("strip_images", po::bool_switch(&strip_images)->default_value(false), "enable stripping of intensity images using the masks")
    ("reorient_images", po::bool_switch(&reorient_images)->default_value(false), "enable reorientation of images to standard coordinate space")
    ("improve_parent", po::bool_switch(&improve_parent)->default_value(true), "enable training constraint of improving parent entropy")
    ;


    // getting the parameters from the command line and configuration file
    po::options_description cmdline_options("options");
    cmdline_options.add(generic).add(config);

    po::options_description config_file_options;
    config_file_options.add(config);

    po::variables_map vm;

    po::store(po::parse_command_line(argc, argv, cmdline_options), vm);
    po::notify(vm);

    if (vm.count("config"))
    {
      std::ifstream ifs(config_file.c_str());
      if (!ifs)
      {
        std::cout << "cannot open config file: " << config_file << std::endl;
        return 0;
      }
      else
      {
        po::store(parse_config_file(ifs, config_file_options), vm);
        po::notify(vm);
      }
    }

    if (vm.count("help") || !vm.count("config"))
    {
      std::cout << cmdline_options << std::endl;
      return 0;
    }
  }
  catch(std::exception& e)
  {
    std::cout << e.what() << std::endl;
    return 1;
  }

  // convert string-based parameters to values
  std::vector<std::string> imagelists;
  strings_to_values(simagelists, imagelists);

  std::vector<float> min_hist_values;
  strings_to_values(smin_hist_values, min_hist_values);

  std::vector<float> max_hist_values;
  strings_to_values(smax_hist_values, max_hist_values);

  std::vector<double> spacing;
  strings_to_values(sspacing, spacing);

  std::vector<double> smoothing;
  strings_to_values(ssmoothing, smoothing);

  // parse lists of filenames
  int channels = imagelists.size();

  std::vector<std::vector<std::string>> image_filenames(channels);
  for (int c = 0; c < channels; c++)
  {
    if (imagelists.size() > 0)
    {
      std::string filename;
      std::vector<std::string> filenames;
      std::ifstream ifs(imagelists[c]);
      while (getline(ifs, filename))
      {
        boost::trim(filename);
        if (filename != "")
        {
          filenames.push_back(filename);
        }
      }
      image_filenames[c] = filenames;
    }
  }

  std::vector<double> labels;
  std::ifstream ifs_labels(labelfile);
  while (ifs_labels.good())
  {
    std::string line;
    getline(ifs_labels, line);
    labels.push_back(atof(line.c_str()));
  }
  ifs_labels.close();

  // generate set of unique labels
  std::set<int> unique_labels;
  for (auto& label : labels)
    unique_labels.insert(label);
  int count_labels = unique_labels.size();


  std::vector<std::string> mask_filenames;
  std::string mask_filename;
  std::ifstream ifs_masks(masklist);
  if (masklist.size() > 0)
  {
    while (getline(ifs_masks, mask_filename))
    {
      boost::trim(mask_filename);
      if (mask_filename != "")
      {
        mask_filenames.push_back(mask_filename);
      }
    }
  }
  bool do_load_masks = mask_filenames.size() == image_filenames[0].size();

  Eigen::Vector3d img_spacing;
  params_to_vector(spacing, img_spacing);

  Eigen::Vector3d img_smoothing;
  params_to_vector(smoothing, img_smoothing);
  bool do_smoothing = img_smoothing.sum() > 0;

  std::vector<std::vector<Image>> intensity_images;
  std::vector<std::vector<Image>> integral_images;
  std::vector<std::vector<IntegralHistogram>> integral_histograms;
  std::vector<Image> masks;

  for (int i = 0; i < image_filenames[0].size(); i++)
  {
    std::cout << "loading data " << i + 1 << " of " << image_filenames[0].size() << " with " << channels << " channel(s)" << std::endl;
    std::vector<Image> intensity_img(channels);
    std::vector<Image> integral_img(channels);
    std::vector<IntegralHistogram> integral_hist(channels);

    // load image and resample
    Image image = itkio::load(image_filenames[0][i]);
    if (reorient_images) image = reorient(image);

    Eigen::Vector3d resample_spacing(img_spacing);
    if (resample_spacing.sum() == 0)
    {
      resample_spacing = image.spacing();
    }

    Image image_resampled = resample(image, resample_spacing, Interpolation::LINEAR);
    image_resampled.dataType(mia::FLOAT); // for temp saving only

    // load sampling mask
    Image mask_resampled = image_resampled.clone();
    if (do_load_masks)
    {
      Image mask = itkio::load(mask_filenames[i]);
      if (reorient_images) mask = reorient(mask);

      resample(mask, mask_resampled, Interpolation::NEAREST);
    }
    else
    {
      ones(mask_resampled);
    }

    // stripping mask
    Image strip_mask = mask_resampled.clone();

    // image smoothing
    if (do_smoothing) gauss(image_resampled, image_resampled, img_smoothing[0], img_smoothing[1], img_smoothing[2]);

    // image stripping
    if (strip_images)
    {
      mul(image_resampled, strip_mask, image_resampled);
    }

    intensity_img[0] = image_resampled;
	if (use_abs_intensity_features || use_diff_intensity_features || use_binary_features)
	{
		integral_img[0] = integral_image(image_resampled);
	}
    if(use_histogram_features) integral_hist[0] = integral_histogram(image_resampled, histogram_bins, min_hist_values[0], max_hist_values[0]);

    // load additional channels and apply same pre-processing
    for (int c = 1; c < channels; c++)
    {
      Image img = itkio::load(image_filenames[c][i]);
      if (reorient_images) img = reorient(img);

      Image img_resampled = image_resampled.clone();
      resample(img, img_resampled, Interpolation::LINEAR);

      if (do_smoothing) gauss(img_resampled, img_resampled, img_smoothing[0], img_smoothing[1], img_smoothing[2]);

      // image stripping
      if (strip_images)
      {
        mul(img_resampled, strip_mask, img_resampled);
      }

      intensity_img[c] = img_resampled;
	  if (use_abs_intensity_features || use_diff_intensity_features || use_binary_features)
	  {
		  integral_img[c] = integral_image(img_resampled);
	  }
      if (use_histogram_features) integral_hist[c] = integral_histogram(img_resampled, histogram_bins, min_hist_values[c], max_hist_values[c]);
    }

    intensity_images.push_back(intensity_img);
    integral_images.push_back(integral_img);
    integral_histograms.push_back(integral_hist);
    masks.push_back(mask_resampled);
  }
  std::cout << "----------------------------------------" << std::endl;

  // checking if data is 2D
  if (!use_2d_features) use_2d_features = intensity_images[0][0].sizeZ() == 1;

  // setup list of feature types
  std::vector<FeatureType> feature_types;
  if (use_point_intensity_features)
  {
    feature_types.push_back(FeatureType::OFFSET_POINT);
    feature_types.push_back(FeatureType::OFFSET_POINT_MINUS_OFFSET_POINT);
  }
  if (use_abs_intensity_features)
  {
    feature_types.push_back(FeatureType::LOCAL_BOX);
    feature_types.push_back(FeatureType::OFFSET_BOX);
  }
  if (use_diff_intensity_features)
  {
    feature_types.push_back(FeatureType::OFFSET_BOX_MINUS_LOCAL_POINT);
    feature_types.push_back(FeatureType::OFFSET_BOX_MINUS_OFFSET_BOX);
  }
  if (use_binary_features)
  {
    feature_types.push_back(FeatureType::OFFSET_BOX_MINUS_LOCAL_POINT_BINARY);
    feature_types.push_back(FeatureType::OFFSET_BOX_MINUS_OFFSET_BOX_BINARY);
  }
  if (use_histogram_features)
  {
    feature_types.push_back(FeatureType::HISTOGRAM_ENTROPY);
    feature_types.push_back(FeatureType::HISTOGRAM_MAX_BIN);
  }

  // setup feature factory
  FeatureFactory feature_factory(feature_types);
  feature_factory.intensity_images(intensity_images);
  feature_factory.integral_images(integral_images);
  feature_factory.integral_histograms(integral_histograms);
  feature_factory.count_features_per_node(count_features_per_node);
  feature_factory.count_thresholds(count_thresholds);
  feature_factory.feature_parameters(channels, min_offset, max_offset, min_boxsize, max_boxsize, use_cross_channel_features, use_2d_features);

  // create output path
  if (!fs::exists(output_path)) fs::create_directories(output_path);

  // write temporary files
  if (write_temp)
  {
    std::stringstream output_temp;
    output_temp << output_path << "/temp";
    if (!fs::exists(output_temp.str())) fs::create_directories(output_temp.str());
    for (int img_index = 0; img_index < intensity_images.size(); img_index++)
    {
      for (int channel = 0; channel < channels; channel++)
      {
        std::stringstream filename_img;
        filename_img << output_path << "/temp/img_" << img_index << "_ch_" << channel << ".nii.gz";
        itkio::save(intensity_images[img_index][channel], filename_img.str());

		//std::stringstream filename_int_img;
		//filename_int_img << output_path << "/temp/img_" << img_index << "_int_" << channel << ".nii.gz";
		//itkio::save(integral_images[img_index][channel], filename_int_img.str());
	  }

      std::stringstream filename_msk;
      filename_msk << output_path << "/temp/img_" << img_index << "_msk.nii.gz";
      itkio::save(masks[img_index], filename_msk.str());
    }
  }

  // check for existing tree files in output path
  std::vector<fs::path> existing_tree_files;
  if (fs::is_directory(output_path))
  {
    fs::recursive_directory_iterator it(output_path);
    fs::recursive_directory_iterator endit;
    while(it != endit)
    {
      if (fs::is_regular_file(*it) && it->path().extension() == ".oak")
      {
        existing_tree_files.push_back(it->path());
      }
      ++it;
    }
  }

  // load/sample set of features
  std::vector<Feature> feature_set;
  std::stringstream filename_feature_set;
  filename_feature_set << output_path << "/features.dat";
  if (fs::exists(filename_feature_set.str()))
  {
    std::cout << "loading features...";
    load(feature_set, filename_feature_set.str());
    count_features_total = feature_set.size();
    std::cout << "done."  << std::endl;
  }
  else
  {
    std::cout << "generating random features...";
    feature_set = feature_factory.sample(count_features_total);
    save(feature_set, filename_feature_set.str());
    std::cout << "done." << std::endl;
  }

  std::cout << "total number of features: " << count_features_total << std::endl;
  std::cout << "----------------------------------------" << std::endl;

  // training trees
  int start_index = existing_tree_files.size();
  int remaining_trees = count_trees-start_index;

  std::cout << "trees: " << remaining_trees << std::endl;
  std::cout << "depth: " << max_depth << std::endl;
  std::cout << "----------------------------------------" << std::endl;

  auto start_forest = std::chrono::high_resolution_clock::now();

  // sampling training data
  std::vector<std::shared_ptr<DataSample<int>>> trainingData;
  for (int img_index = 0; img_index < intensity_images.size(); img_index++)
  {
    Image img_resampled = intensity_images[img_index][0];
    int x = static_cast<int>((img_resampled.sizeX() - 1) / 2.0 + 0.5);
    int y = static_cast<int>((img_resampled.sizeY() - 1) / 2.0 + 0.5);
    int z = static_cast<int>((img_resampled.sizeZ() - 1) / 2.0 + 0.5);

    std::shared_ptr<DataSample<int>> sample = std::make_shared<DataSample<int>>();
    sample->image_index = img_index;
    sample->label = static_cast<int>(labels[img_index]);
    sample->point = Eigen::Vector3d(x, y, z);
    trainingData.push_back(sample);
  }

  std::vector<DataSample<int>*> trainingSamples;
  for (auto& sample : trainingData)
    trainingSamples.push_back(sample.get());

  auto feature_map = Image(intensity_images[0][0]).clone();
  feature_map.dataType(mia::FLOAT);
  zeros(feature_map);

  auto feature_map_all = Image(intensity_images[0][0]).clone();
  feature_map_all.dataType(mia::FLOAT);
  zeros(feature_map_all);
  for (const auto &feature : feature_set)
  {
    update_feature_map(feature, trainingSamples[0]->point, feature_map_all);
  }

  for (int tree = 0; tree < remaining_trees; tree++)
  {
    Node<Classifier> root;

    std::cout << "training tree " << tree+1 << " of " << remaining_trees << " with " << trainingSamples.size() << " samples" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    auto start_tree = std::chrono::high_resolution_clock::now();

    train_tree(root, trainingSamples, feature_factory, feature_set, count_labels, max_depth, min_samples, improve_parent);

    auto stop_tree = std::chrono::high_resolution_clock::now();
    std::cout << "++ finished in " << std::chrono::duration_cast< std::chrono::milliseconds >(stop_tree-start_tree).count() << "ms" << std::endl;

    std::cout << "++ saving tree...";

    std::stringstream filename;
    filename << output_path << "/tree_" << tree + start_index + 1 << ".oak";
    save(root, filename.str());

    std::vector<int> leaf_indices;
    for (const auto sample : trainingSamples)
    {
      auto path_and_leaf_index = descend_tree_path_node_index(root, *sample, feature_factory, feature_set, max_depth);
      leaf_indices.push_back(path_and_leaf_index.second);
      for (int n = 0; n < path_and_leaf_index.first.size() - 1; n++)
      {
        auto feature = feature_set[path_and_leaf_index.first[n]->split_function.feature_index];
        update_feature_map(feature, trainingSamples[0]->point, feature_map);
      }
    }

    std::stringstream filename_leaf_idx;
    filename_leaf_idx << output_path << "/tree_" << tree + start_index + 1 << ".idx";
    std::ofstream ofs_leaf_idx(filename_leaf_idx.str());
    for (auto leaf_index : leaf_indices)
    {
      ofs_leaf_idx << leaf_index << std::endl;
    }
    ofs_leaf_idx.close();

    std::cout << "done." << std::endl;
    std::cout << "----------------------------------------" << std::endl;
  }

  std::stringstream filename_feature_map;
  filename_feature_map << output_path << "/feature_map.nii.gz";
  itkio::save(feature_map, filename_feature_map.str());

  if (do_smoothing)
  {
    gauss(feature_map, feature_map, img_smoothing[0], img_smoothing[1], img_smoothing[2]);
    std::stringstream filename_feature_map_smooth;
    filename_feature_map_smooth << output_path << "/feature_map_smooth.nii.gz";
    itkio::save(feature_map, filename_feature_map_smooth.str());
  }

  std::stringstream filename_feature_map_all;
  filename_feature_map_all << output_path << "/feature_map_all.nii.gz";
  itkio::save(feature_map_all, filename_feature_map_all.str());

  auto stop_forest = std::chrono::high_resolution_clock::now();
  std::cout << "++ training took " << std::chrono::duration_cast< std::chrono::milliseconds >(stop_forest-start_forest).count() << "ms" << std::endl;
  std::cout << "----------------------------------------" << std::endl;
}
