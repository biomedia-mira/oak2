# oak2

[![Build Status](https://travis-ci.org/biomedia-mira/oak2.svg?branch=master)](https://travis-ci.org/biomedia-mira/oak2)

*oak2* is a cross-platform C++ implementation of Random Forests supporting local and global image classification and regression. Local prediction models can be employed for problems such as image segmentation and image synthesis, while global models enable image-level predictions.

## Dependencies

*oak2* depends on several third-party libraries:

* [Eigen](eigen.tuxfamily.org)
* [Intel TBB](https://www.threadingbuildingblocks.org/) (tested up to v.4.4)
* [Boost](http://www.boost.org/) (tested up to v1.58)
* [ITK](http://itk.org) (tested up to [v4.13.2](https://sourceforge.net/projects/itk/files/itk/4.13/InsightToolkit-4.13.2.tar.gz))

## Build instructions

Eigen is a header-only library and can be simply installed via:

```
#!bash

$ mkdir 3rdparty
$ cd 3rdparty
$ wget https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz --progress=bar:force:noscroll
$ mkdir eigen
$ tar xf eigen-3.3.7.tar.gz -C eigen --strip-components=1
```

You can download and install ITK in the same `3rdparty` folder via:

```
#!bash

$ wget https://sourceforge.net/projects/itk/files/itk/5.0/InsightToolkit-5.0.0.tar.gz
$ tar xvf InsightToolkit-5.0.0.tar.gz
$ cd InsightToolkit-5.0.0
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX=../../itk ..
$ make -j4
$ make install
```

Alternatively, you can check out these [ITK install instructions](https://itk.org/Wiki/ITK/Getting_Started/Build/Linux).

You can install Boost and TBB via `apt-get`:

```
#!bash

$ sudo apt-get install libboost-all-dev libtbb-dev
```

Note, you might have to specify a specific version via `apt-get install <package>=<version>`.

*oak2* comes with a CMake configuration file. From the top folder where `CMakeLists.txt` is located (same as this README), do the following to build all internal libraries and executables:

```
#!bash

$ mkdir build
$ cd build
$ export THIRD_PARTY_DIR=<folder_containing_eigen_and_itk>
$ cmake ..
$ make -j4

```

The command line tools are located in `build/oak/apps`.

## Usage

Detailed instructions and examples are coming soon...

## Acknowledgements

Special thanks go to [Hauke Heibel](https://github.com/hauke76) who has contributed significantly to the C++ implementation.
