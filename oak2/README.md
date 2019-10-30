# oak2 library #

oak2 is a cross-platform C++ library for random forests. it includes functionality for classification and multi-variate regression.

## Dependencies ##

oak2 depends on several libraries which need to be installed.

* [Mia library](https://bitbucket.org/bglocker/mia)
* [Intel TBB](https://www.threadingbuildingblocks.org/)
* [Boost](http://www.boost.org/) (component [serialization](http://www.boost.org/doc/libs/1_57_0/libs/serialization/doc/index.html))

## Build instructions ##

oak2 comes with a CMake configuration file. In order to be able to find the mia library, you should have both mia and oak2 under the same parent folder. In this parent folder, you need to create a CMakeLists.txt file containing the following lines:

```
#!text

cmake_minimum_required(VERSION 2.8.12)

project(common)

add_subdirectory(mia)
add_subdirectory(oak2)

```

From the folder where the CMakeLists.txt is located, you can then do the following to build oak2 and mia:

```
#!bash

$ mkdir build
$ cd build
$ cmake ..
$ make

```

## Build everything ##

In order to build the binary tools for training and testing, add the apps subdirectory and the [itkio](https://bitbucket.org/bglocker/itkio) library to the CMakeLists.txt.

```
#!text

cmake_minimum_required(VERSION 2.8.12)

project(common)

add_subdirectory(mia)
add_subdirectory(itkio)
add_subdirectory(oak2)
add_subdirectory(oak2/apps)

```

Make sure the all dependencies are installed, and download and install [ITK](http://itk.org) and [Eigen](http://eigen.tuxfamily.org):

```
sudo apt-get install libboost-all-dev libtbb-dev
export THIRD_PARTY_DIR=<folder_containing_eigen>

```

For [Eigen](http://eigen.tuxfamily.org) and [ITK](http://itk.org) make sure to rename the top folders after unzipping/installing to eigen and itk.