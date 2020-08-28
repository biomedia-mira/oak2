FROM ubuntu:18.04

RUN apt-get update && \
	apt-get install -y build-essential git cmake autoconf libtool pkg-config wget


RUN mkdir -p /oak2/3rdparty 

WORKDIR /oak2/3rdparty

ADD https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz . 
RUN mkdir eigen && tar xf eigen-3.3.7.tar.gz -C eigen --strip-components=1

ADD https://github.com/InsightSoftwareConsortium/ITK/releases/download/v5.1.0/InsightToolkit-5.1.0.tar.gz . 
RUN tar xvf InsightToolkit-5.1.0.tar.gz && cd InsightToolkit-5.1.0 && mkdir build && \
 cd build && cmake -DCMAKE_INSTALL_PREFIX=../../itk .. && make -j4 && make install

RUN apt-get install -y libboost-all-dev libtbb-dev 

COPY . /oak2

WORKDIR /oak2

ENV THIRD_PARTY_DIR=/oak2/3rdparty

RUN mkdir build && cd build && cmake .. && make -j4

RUN find -type f -executable -exec cp {} /usr/local/bin/ \;