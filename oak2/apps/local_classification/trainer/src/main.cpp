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

int main(int argc, char *argv[])
{
  std::string config_file;

  std::vector<std::string> simages;
  std::vector<std::string> simagelists;
  std::vector<std::string> slabels;
  std::vector<std::string> smin_hist_values;
  std::vector<std::string> smax_hist_values;
  std::vector<std::string> sspacing;
  std::vector<std::string> ssmoothing;
  std::vector<std::string> slower_thresholds;
  std::vector<std::string> supper_thresholds;
  std::string labellist;
  std::string labelmap("");
  std::string masklist;
  std::string imagemask("");
  std::string output_path;

  bool write_temp = false;

  int count_trees = 0;
  int max_depth = 0;
  int min_samples = 0;
  int count_features_total = 0;
  int count_features_per_node = 0;
  int count_thresholds = 0;
  bool use_point_intensity_features = false;
  bool use_abs_intensity_features = true;
  bool use_diff_intensity_features = true;
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
  bool background_sampling_only = false;
  int background_boundary = 0;
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
    ("image", po::value<std::vector<std::string>>(&simages)->multitoken(), "filename(s) of single training image")
    ("labelmap", po::value<std::string>(&labelmap), "filename of label map of single training image")
    ("mask", po::value<std::string>(&imagemask), "filename of sampling mask of single training image")
    ("images", po::value<std::vector<std::string>>(&simagelists)->multitoken(), "text file(s) listing images")
    ("labelmaps", po::value<std::string>(&labellist), "text file listing label maps")
    ("masks", po::value<std::string>(&masklist), "text file listing sampling masks")
    ("labels", po::value<std::vector<std::string>>(&slabels)->multitoken(), "list of labels")
    ("trees", po::value<int>(&count_trees), "number of trees")
    ("depth", po::value<int>(&max_depth), "maxmimum tree depth")
    ("min_samples", po::value<int>(&min_samples), "minimum number of samples in leaf nodes")
    ("features_total", po::value<int>(&count_features_total), "number of total random features")
    ("features_per_node", po::value<int>(&count_features_per_node), "number of random features tested per node")
    ("thresholds", po::value<int>(&count_thresholds), "number of thresholds per feature")
    ("point_intensity_features", po::bool_switch(&use_point_intensity_features)->default_value(false), "enable point intensity features")
    ("abs_intensity_features", po::bool_switch(&use_abs_intensity_features)->default_value(true), "enable absolute intensity features")
    ("diff_intensity_features", po::bool_switch(&use_diff_intensity_features)->default_value(true), "enable intensity difference features")
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
    ("lower_intensity_thresholds", po::value<std::vector<std::string>>(&slower_thresholds)->multitoken(), "lower intensity threshold(s)")
    ("upper_intensity_thresholds", po::value<std::vector<std::string>>(&supper_thresholds)->multitoken(), "upper intensity threshold(s)")
    ("strip_images", po::bool_switch(&strip_images)->default_value(false), "enable stripping of intensity images using the masks")
    ("reorient_images", po::bool_switch(&reorient_images)->default_value(false), "enable reorientation of images to standard coordinate space")
    ("improve_parent", po::bool_switch(&improve_parent)->default_value(true), "enable training constraint of improving parent entropy")
    ("sampling_rate", po::value<float>(&data_sampling_rate), "sampling probability for bagging of training data")
    ("background_sampling_only", po::bool_switch(&background_sampling_only)->default_value(false), "enable sampling on background class only")
    ("background_boundary", po::value<int>(&background_boundary)->default_value(0), "number of dilations used for generating a background boundary")
    ;

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
  std::vector<std::string> images;
  strings_to_values(simages, images);

  std::vector<std::string> imagelists;
  strings_to_values(simagelists, imagelists);

  std::vector<int> labels;
  strings_to_values(slabels, labels);

  std::vector<float> min_hist_values;
  strings_to_values(smin_hist_values, min_hist_values);

  std::vector<float> max_hist_values;
  strings_to_values(smax_hist_values, max_hist_values);

  std::vector<double> spacing;
  strings_to_values(sspacing, spacing);

  std::vector<double> smoothing;
  strings_to_values(ssmoothing, smoothing);

  std::vector<float> lower_thresholds;
  strings_to_values(slower_thresholds, lower_thresholds);

  std::vector<float> upper_thresholds;
  strings_to_values(supper_thresholds, upper_thresholds);

  // generate set of unique labels and label mapping
  int id = 0;
  std::set<int> unique_labels;
  for (auto& label : labels)
    unique_labels.insert(label);
  std::map<float, int> label_transform;
  std::vector<int> original_labels(unique_labels.size());
  for (auto& label : unique_labels)
  {
    original_labels[id] = label;
    label_transform.insert(std::pair<float, int>(label, id++));
  }

  int count_labels = unique_labels.size();

  // parse lists of filenames
  int channels = std::max(imagelists.size(),images.size());

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
    if (images.size() == channels) image_filenames[c].push_back(images[c]);
  }

  std::vector<std::string> label_filenames;
  std::string label_filename;
  std::ifstream ifs_labelmaps(labellist);
  if (labellist.size() > 0)
  {
    while (getline(ifs_labelmaps, label_filename))
    {
      boost::trim(label_filename);
      if (label_filename != "")
      {
        label_filenames.push_back(label_filename);
      }
    }
  }
  if (labelmap != "") label_filenames.push_back(labelmap);

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
  if (imagemask != "") mask_filenames.push_back(imagemask);
  bool do_load_masks = mask_filenames.size() == image_filenames[0].size();

  Eigen::Vector3d img_spacing;
  params_to_vector(spacing, img_spacing);

  Eigen::Vector3d img_smoothing;
  params_to_vector(smoothing, img_smoothing);
  bool do_smoothing = img_smoothing.sum() > 0;

  std::vector<std::vector<Image>> intensity_images;
  std::vector<std::vector<Image>> integral_images;
  std::vector<std::vector<IntegralHistogram>> integral_histograms;
  std::vector<Image> labelmaps;
  std::vector<Image> boundaries;
  std::vector<Image> masks;

  for (int i = 0; i < label_filenames.size(); i++)
  {
    std::cout << "loading data " << i+1 << " of " << label_filenames.size() << " with " << channels << " channel(s)" << std::endl;
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

    // generate valid sampling mask based on lower and upper intensity thresholds
    Image mask_thres = image.clone();
    float lower = lower_thresholds.size() > 0 ? lower_thresholds[0] : std::numeric_limits<float>::lowest();
    float upper = upper_thresholds.size() > 0 ? upper_thresholds[0] : std::numeric_limits<float>::max();
    threshold(image, mask_thres, lower, upper);
    Image mask_thres_resampled = image_resampled.clone();
    resample(mask_thres, mask_thres_resampled, Interpolation::NEAREST);
    mul(mask_resampled, mask_thres_resampled, mask_resampled);

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

      Image msk_thres = img.clone();
      float lower = lower_thresholds.size() > c ? lower_thresholds[c] : std::numeric_limits<float>::lowest();
      float upper = upper_thresholds.size() > c ? upper_thresholds[c] : std::numeric_limits<float>::max();
      threshold(img, msk_thres, lower, upper);
      Image msk_thres_resampled = image_resampled.clone();
      resample(msk_thres, msk_thres_resampled, Interpolation::NEAREST);
      mul(mask_resampled, msk_thres_resampled, mask_resampled);

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

    // load label map
    Image lmap = itkio::load(label_filenames[i]);
    if (reorient_images) lmap = reorient(lmap);

    Image labelmap_resampled = image_resampled.clone();
    resample(lmap, labelmap_resampled, Interpolation::NEAREST);

    Image boundary = image_resampled.clone();
    zeros(boundary);

    if (background_boundary > 0)
    {
      threshold(labelmap_resampled, boundary, labels[0], labels[0]);
      invert_binary(boundary, boundary);
      Image boundary_dilated = boundary.clone();
      for (int i = 0; i < background_boundary; i++)
      {
        dilate_binary(boundary_dilated.clone(), boundary_dilated);
      }
      sub(boundary_dilated, boundary, boundary);
    }

    intensity_images.push_back(intensity_img);
    integral_images.push_back(integral_img);
    integral_histograms.push_back(integral_hist);
    labelmaps.push_back(labelmap_resampled);
    boundaries.push_back(boundary);
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

      std::stringstream filename_lab;
      filename_lab << output_path << "/temp/img_" << img_index << "_lab.nii.gz";
      itkio::save(labelmaps[img_index], filename_lab.str());

      std::stringstream filename_bnd;
      filename_bnd << output_path << "/temp/img_" << img_index << "_bnd.nii.gz";
      itkio::save(boundaries[img_index], filename_bnd.str());

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
  for (int tree = 0; tree < remaining_trees; tree++)
  {
    // sampling training data
    std::vector<std::shared_ptr<DataSample<int>>> trainingData;
    for (int img_index = 0; img_index < intensity_images.size(); img_index++)
    {
      Image img_resampled = intensity_images[img_index][0];
      Image sampling_mask = img_resampled.clone();
      if (data_sampling_rate > 0) random_binary(sampling_mask, data_sampling_rate);
      else ones(sampling_mask);
      mul(sampling_mask, masks[img_index], sampling_mask);
      for (int z = 0; z < img_resampled.sizeZ(); z++)
      {
        for (int y = 0; y < img_resampled.sizeY(); y++)
        {
          for (int x = 0; x < img_resampled.sizeX(); x++)
          {
            int class_label = 0; //default set to background for all unspecified labels
            auto it = label_transform.find(labelmaps[img_index](x, y, z));
            if (it != label_transform.end())
            {
              class_label = it->second;
            }

            // sample training point
            if (sampling_mask(x,y,z) != 0.0f
                || (background_sampling_only && masks[img_index](x, y, z) != 0 && class_label != 0)
                || (background_boundary > 0 && boundaries[img_index](x,y,z) != 0 && masks[img_index](x,y,z) != 0))
            {
              std::shared_ptr<DataSample<int>> sample = std::make_shared<DataSample<int>>();
              sample->image_index = img_index;
              sample->label = class_label;
              sample->point = Eigen::Vector3d(x,y,z);
              trainingData.push_back(sample);
            }
          }
        }
      }
    }
    std::vector<DataSample<int>*> trainingSamples;
    for (auto& sample : trainingData)
      trainingSamples.push_back(sample.get());

    Node<Classifier> root;

    std::cout << "training tree " << tree+1 << " of " << remaining_trees << " with " << trainingSamples.size() << " samples" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    auto start_tree = std::chrono::high_resolution_clock::now();

    train_tree(root, trainingSamples, feature_factory, feature_set, count_labels, max_depth, min_samples, improve_parent);

    auto stop_tree = std::chrono::high_resolution_clock::now();
    std::cout << "++ finished in " << std::chrono::duration_cast< std::chrono::milliseconds >(stop_tree-start_tree).count() << "ms" << std::endl;

    std::cout << "++ saving tree...";

    std::stringstream filename;
    filename << output_path << "/tree_" << tree+start_index+1 << ".oak";
    save(root, filename.str());

    std::cout << "done." << std::endl;
    std::cout << "----------------------------------------" << std::endl;
  }

  auto stop_forest = std::chrono::high_resolution_clock::now();
  std::cout << "++ training took " << std::chrono::duration_cast< std::chrono::milliseconds >(stop_forest-start_forest).count() << "ms" << std::endl;
  std::cout << "----------------------------------------" << std::endl;
}
