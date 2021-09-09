ARG PYTORCH="1.8.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak
RUN sed -i s:/archive.ubuntu.com:/mirrors.tuna.tsinghua.edu.cn/ubuntu:g /etc/apt/sources.list
RUN cat /etc/apt/sources.list
RUN apt-get clean && apt-get update && apt-get install -y --fix-missing git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 ffmpeg openssh-server\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install mmcv-full
RUN pip install mmcv-full==latest+torch1.8.0+cu111 -f https://download.openmmlab.com/mmcv/dist/index.html

# Install MMAction2
RUN conda clean --all
RUN git clone git://github.com/open-mmlab/mmaction2.git /home/wangjiahao/workspace/mmaction2
WORKDIR /home/wangjiahao/workspace/mmaction2
RUN mkdir -p /mmaction2/data
ENV FORCE_CUDA="1"
RUN pip install cython --no-cache-dir
RUN pip install --no-cache-dir -e .
