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
#include <tbb/parallel_for.h>

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
  std::vector<std::string> smin_hist_values;
  std::vector<std::string> smax_hist_values;
  std::vector<std::string> sspacing;
  std::vector<std::string> ssmoothing;
  std::string masklist;
  std::string namelist;
  std::string imagemask("");
  std::string outputname("");
  std::string forest_path;
  std::string output_path;

  bool write_temp = false;

  int count_trees = 0;
  int max_depth = 0;
  int histogram_bins = 0;
  bool use_histogram_features = false;
  bool strip_images = false;
  bool reorient_images = false;
  bool use_folder_name = false;

  int label_dim = 1;

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
    ("forest", po::value<std::string>(&forest_path), "forest path")
    ("image", po::value<std::vector<std::string>>(&simages)->multitoken(), "filename(s) of single test image")
    ("mask", po::value<std::string>(&imagemask), "filename of sampling mask of single test image")
	("name", po::value<std::string>(&outputname), "basename for prediction files")
    ("images", po::value<std::vector<std::string>>(&simagelists)->multitoken(), "text file(s) listing images")
    ("masks", po::value<std::string>(&masklist), "text file listing sampling masks")
	("names", po::value<std::string>(&namelist), "text file listing output names")
    ("trees", po::value<int>(&count_trees), "number of trees")
    ("depth", po::value<int>(&max_depth), "maxmimum tree depth")
    ("histogram_features", po::bool_switch(&use_histogram_features)->default_value(false), "enable histogram features")
    ("histogram_bins", po::value<int>(&histogram_bins), "number of bins for histogram features")
    ("histogram_min_values", po::value<std::vector<std::string>>(&smin_hist_values)->multitoken(), "minimum intensity value(s) for histogram features")
    ("histogram_max_values", po::value<std::vector<std::string>>(&smax_hist_values)->multitoken(), "maximum intensity value(s) for histogram features")
    ("spacing", po::value<std::vector<std::string>>(&sspacing)->multitoken(), "element spacing for image resampling")
    ("smoothing", po::value<std::vector<std::string>>(&ssmoothing)->multitoken(), "sigmas for Gaussian image smoothing")
    ("strip_images", po::bool_switch(&strip_images)->default_value(false), "enable stripping of intensity images using the masks")
    ("reorient_images", po::bool_switch(&reorient_images)->default_value(false), "enable reorientation of images to standard coordinate space")
    ("use_folder_name", po::bool_switch(&use_folder_name)->default_value(false), "use the parent folder name for output files")
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

  std::vector<float> min_hist_values;
  strings_to_values(smin_hist_values, min_hist_values);

  std::vector<float> max_hist_values;
  strings_to_values(smax_hist_values, max_hist_values);

  std::vector<double> spacing;
  strings_to_values(sspacing, spacing);

  std::vector<double> smoothing;
  strings_to_values(ssmoothing, smoothing);

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

  std::vector<std::string> output_filenames;
  std::string output_filename;
  std::ifstream ifs_names(namelist);
  if (namelist.size() > 0)
  {
	  while (getline(ifs_names, output_filename))
	  {
		  boost::trim(output_filename);
		  if (output_filename != "")
		  {
			  output_filenames.push_back(output_filename);
		  }
	  }
  }
  if (outputname != "") output_filenames.push_back(outputname);

  Eigen::Vector3d img_spacing;
  params_to_vector(spacing, img_spacing);

  Eigen::Vector3d img_smoothing;
  params_to_vector(smoothing, img_smoothing);
  bool do_smoothing = img_smoothing.sum() > 0;

  // load set of features
  std::vector<Feature> feature_set;
  std::stringstream filename_feature_set;
  filename_feature_set << forest_path << "/features.dat";
  if (fs::exists(filename_feature_set.str()))
  {
    std::cout << "loading features...";
    load(feature_set, filename_feature_set.str());
    std::cout << "done."  << std::endl;
  }

  std::cout << "total number of features: " << feature_set.size() << std::endl;
  std::cout << "----------------------------------------" << std::endl;

  // loading trees
  std::vector<fs::path> existing_tree_files;
  if (fs::is_directory(forest_path))
  {
    fs::recursive_directory_iterator it(forest_path);
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

  if (max_depth == 0) max_depth = std::numeric_limits<int>::max();
  if (count_trees == 0) count_trees = std::numeric_limits<int>::max();
  count_trees = std::min(count_trees, static_cast<int>(existing_tree_files.size()));

  std::string depth_str = max_depth == std::numeric_limits<int>::max() ? "max" : std::to_string(max_depth);
  std::cout << "-- trees: " << count_trees << std::endl;
  std::cout << "-- depth: " << depth_str << std::endl;
  std::cout << "----------------------------------------" << std::endl;

  std::cout << "loading " << count_trees  << " trees" << std::endl;
  std::cout << "----------------------------------------" << std::endl;
  std::vector<std::shared_ptr<Node<Regressor>>> trees;
  for (int tree = 0; tree < count_trees; tree++)
  {
    std::shared_ptr<Node<Regressor>> root = std::make_shared<Node<Regressor>>();
    load(*root, existing_tree_files[tree].generic_string());
    trees.push_back(root);
  }

  if (!fs::exists(output_path)) fs::create_directories(output_path);

  int count_images = image_filenames[0].size();
  if (!use_folder_name && count_images > 1)
  {
    auto basename1 = fs::basename(image_filenames[0][0]);
    auto basename2 = fs::basename(image_filenames[0][1]);
    use_folder_name = basename1 == basename2;
  }
  int folder_level = 0;
  if (use_folder_name)
  {
    fs::path input_path1(image_filenames[0][0]);
    fs::path input_path2(image_filenames[0][1]);
    while (fs::basename(input_path1.parent_path()) == fs::basename(input_path2.parent_path()))
    {
      input_path1 = input_path1.parent_path();
      input_path2 = input_path2.parent_path();
      folder_level++;
    }
  }
  for (int img_index = 0; img_index < count_images; img_index++)
  {
    std::cout << "-- testing image " << img_index+1 << " of " << count_images << "...";

    std::vector<std::vector<Image>> intensity_images;
    std::vector<std::vector<Image>> integral_images;
    std::vector<std::vector<IntegralHistogram>> integral_histograms;

    std::vector<Image> intensity_img(channels);
    std::vector<Image> integral_img(channels);
    std::vector<IntegralHistogram> integral_hist(channels);

    // load image and resample
    Image image = itkio::load(image_filenames[0][img_index]);
    if (reorient_images) image = reorient(image);

    Eigen::Vector3d resample_spacing(img_spacing);
    if (resample_spacing.sum() == 0)
    {
      resample_spacing = image.spacing();
    }

    Image image_resampled = resample(image, resample_spacing, Interpolation::LINEAR);

    // load sampling mask
    Image mask_resampled = image_resampled.clone();
    if (do_load_masks)
    {
      Image mask = itkio::load(mask_filenames[img_index]);
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
    integral_img[0] = integral_image(image_resampled);
    if(use_histogram_features) integral_hist[0] = integral_histogram(image_resampled, histogram_bins, min_hist_values[0], max_hist_values[0]);

    // load additional channels and apply same pre-processing
    for (int c = 1; c < channels; c++)
    {
      Image img = itkio::load(image_filenames[c][img_index]);
      if (reorient_images) img = reorient(img);

      Image img_resampled = intensity_img[0].clone();
      resample(img, img_resampled, Interpolation::LINEAR);

      if (do_smoothing) gauss(img_resampled, img_resampled, img_smoothing[0], img_smoothing[1], img_smoothing[2]);

      // image stripping
      if (strip_images)
      {
        mul(img_resampled, strip_mask, img_resampled);
      }

      intensity_img[c] = img_resampled;
      integral_img[c] = integral_image(img_resampled);
      if (use_histogram_features) integral_hist[c] = integral_histogram(img_resampled, histogram_bins, min_hist_values[c], max_hist_values[c]);
    }

    intensity_images.push_back(intensity_img);
    integral_images.push_back(integral_img);
    integral_histograms.push_back(integral_hist);

    FeatureFactory feature_factory;
    feature_factory.intensity_images(intensity_images);
    feature_factory.integral_images(integral_images);
    feature_factory.integral_histograms(integral_histograms);

    // write temporary files
    if (write_temp)
    {
      std::stringstream output_temp;
      output_temp << output_path << "/temp";
      if (!fs::exists(output_temp.str())) fs::create_directories(output_temp.str());
      for (int channel = 0; channel < channels; channel++)
      {
        std::stringstream filename_img;
        filename_img << output_path << "/temp/img_" << img_index << "_ch_" << channel << ".nii.gz";
        itkio::save(intensity_img[channel], filename_img.str());
      }

      std::stringstream filename_msk;
      filename_msk << output_path << "/temp/img_" << img_index << "_msk.nii.gz";
      itkio::save(mask_resampled, filename_msk.str());
    }

    // testing output
    std::vector<double> prediction_mean(label_dim);
    std::vector<double> prediction_var(label_dim);
    for (size_t i = 0; i < label_dim; i++)
    {
      prediction_mean[i] = 0.0;
      prediction_var[i] = 0.0;
    }

    // testing trees
    auto start_testing = std::chrono::high_resolution_clock::now();

    int x = static_cast<int>((image_resampled.sizeX() - 1) / 2.0 + 0.5);
    int y = static_cast<int>((image_resampled.sizeY() - 1) / 2.0 + 0.5);
    int z = static_cast<int>((image_resampled.sizeZ() - 1) / 2.0 + 0.5);

    DataSample<int> sample;
    sample.image_index = 0;
    sample.point = Eigen::Vector3d(x, y, z);

    std::vector<int> leaf_indices;
    for (const auto& tree : trees)
    {
      auto node = descend_tree(*tree, sample, feature_factory, feature_set, max_depth);
      auto index = node_index(*tree, sample, feature_factory, feature_set, max_depth);
      for (int c = 0; c < label_dim; c++)
      {
        prediction_mean[c] += static_cast<double>(node.predictor.means[c] / static_cast<double>(count_trees));
        prediction_var[c] += static_cast<double>(node.predictor.variances[c] / static_cast<double>(count_trees));
      }
      leaf_indices.push_back(index);
    }

    auto stop_testing = std::chrono::high_resolution_clock::now();
    std::cout << "took " << std::chrono::duration_cast< std::chrono::milliseconds >(stop_testing-start_testing).count() << "ms" << std::endl;

    // saving results
    fs::path input_path(image_filenames[0][img_index]);
    std::string basename = fs::basename(input_path);

	if (output_filenames.size() > img_index)
	{
		basename = fs::basename(output_filenames[img_index]);
	}
	else
	{
		if (use_folder_name)
		{
			for (int i = 0; i < folder_level; i++)
			{
				input_path = input_path.parent_path();
			}
			basename = fs::basename(input_path.parent_path());
		}
	}

    if (fs::extension(basename) != "") basename = fs::basename(basename);
    std::stringstream filename;
    filename << output_path << "/" << basename << "_prediction.txt";

    std::ofstream ofs(filename.str());
    for (int c = 0; c < label_dim; c++)
    {
      ofs << prediction_mean[c] << "\t" << prediction_var[c] << std::endl;
    }
    ofs.close();

    std::stringstream filename_leaf_idx;
    filename_leaf_idx << output_path << "/" << basename << "_leaf_indices.txt";
    std::ofstream ofs_leaf_idx(filename_leaf_idx.str());
    for (auto leaf_index : leaf_indices)
    {
      ofs_leaf_idx << leaf_index << std::endl;
    }
    ofs_leaf_idx.close();
  }
}
