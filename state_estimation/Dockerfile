# =============================================================================
# OpenVINS VIO Docker Image
# Multi-stage build: ROS2 Humble + OpenVINS v2.7 + evaluation tools
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Build everything
# ---------------------------------------------------------------------------
FROM ros:humble-ros-base-jammy AS builder

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
# libceres-dev pulls in: libeigen3-dev, libgflags-dev, libgoogle-glog-dev, libsuitesparse-dev
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    git \
    libatlas-base-dev \
    libboost-all-dev \
    libceres-dev \
    libeigen3-dev \
    libopencv-dev \
    python3-dev \
    python3-pip \
  && rm -rf /var/lib/apt/lists/*

# Python evaluation and visualization tools
RUN pip3 install --no-cache-dir \
    evo \
    numpy \
    rerun-sdk

# Clone OpenVINS v2.7
RUN mkdir -p /catkin_ws/src && \
    git clone --branch v2.7 --depth 1 \
        https://github.com/rpng/open_vins.git \
        /catkin_ws/src/open_vins

# Install remaining ROS dependencies (cv_bridge, image_transport, etc.)
RUN apt-get update && \
    cd /catkin_ws && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y && \
    rm -rf /var/lib/apt/lists/*

# Build OpenVINS packages
# MAKEFLAGS=-j2 limits cmake parallelism to avoid OOM on memory-constrained Docker hosts.
SHELL ["/bin/bash", "-c"]
ENV MAKEFLAGS="-j2"
RUN source /opt/ros/humble/setup.bash && \
    cd /catkin_ws && \
    colcon build \
      --event-handlers console_cohesion+ \
      --packages-select ov_core ov_init ov_msckf ov_eval \
      --parallel-workers 1 \
      --cmake-args -DCMAKE_BUILD_TYPE=Release

# ---------------------------------------------------------------------------
# Stage 2: Runtime (lean image)
# ---------------------------------------------------------------------------
FROM ros:humble-ros-base-jammy AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Runtime-only system libraries — no -dev packages, no separate ros-humble-* apt installs.
# cv_bridge, image_transport, and opencv libs are copied from builder instead.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libatlas3-base \
    libboost-filesystem1.74.0 \
    libboost-system1.74.0 \
    libceres2 \
    libgflags2.2 \
    libgoogle-glog0v5 \
    libopencv-core4.5d \
    libopencv-imgproc4.5d \
    libopencv-highgui4.5d \
    libopencv-features2d4.5d \
    libopencv-calib3d4.5d \
    libopencv-imgcodecs4.5d \
    libopencv-contrib4.5d \
    libopencv-video4.5d \
    libopencv-objdetect4.5d \
    libopencv-flann4.5d \
    libsuitesparseconfig5 \
  && rm -rf /var/lib/apt/lists/*

# Copy ROS2 packages installed by rosdep (cv_bridge, image_transport) from builder
# instead of reinstalling via apt to avoid pulling hundreds of MB of transitive deps
COPY --from=builder /opt/ros/humble /opt/ros/humble

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy built OpenVINS install space and config files
COPY --from=builder /catkin_ws/install /catkin_ws/install
COPY --from=builder /catkin_ws/src/open_vins/config /catkin_ws/config

# Source ROS2 and OpenVINS in every shell
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /catkin_ws/install/setup.bash" >> /root/.bashrc

WORKDIR /catkin_ws

SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source /catkin_ws/install/setup.bash && exec \"$@\"", "--"]
CMD ["/bin/bash"]
