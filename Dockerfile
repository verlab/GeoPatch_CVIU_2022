# Base image
FROM tensorflow/tensorflow:1.15.5-py3

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    pkg-config \
    wget \
    unzip && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /src

#Install OpenCV 3.4.8
RUN wget "https://github.com/opencv/opencv/archive/3.4.8.zip" && \
    unzip 3.4.8.zip && \
    mkdir -p opencv-3.4.8/build && \
    cd opencv-3.4.8/build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_LIST=core,highgui,imgproc,features2d .. && \
    make -j$(nproc) && \
    make install


# Download and extract SimulationICCV.tar.gz
RUN wget "https://www.verlab.dcc.ufmg.br/nonrigid/SimulationICCV.tar.gz" && \
    tar -xvf SimulationICCV.tar.gz

RUN pip3 install opencv-python-headless==3.4.3.18 scipy --prefer-binary

# Copy geopatch source from the current directory into the container
COPY . /src

# Build geopatch-forge/geopatch
RUN cd /src/geopatch && \
    mkdir -p build && cd build && \
    cmake .. && make -j$(nproc)

# Clean up temporary files to reduce image size
RUN rm -rf /src/opencv-3.4.8 /src/3.4.8.zip /src/SimulationICCV.tar.gz

# Default command
CMD ["/bin/bash"]

