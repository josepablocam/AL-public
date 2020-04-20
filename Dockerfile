FROM continuumio/miniconda

ARG VERSION=0.1

LABEL container.base.image="python:3.6.1"
LABEL software.name="AL"
LABEL software.version=${VERSION}
LABEL software.description="Container for demo version of AL"


# RUN echo "deb http://archive.debian.org/debian jessie-backports main" >> /etc/apt/sources.list
# RUN echo 'Acquire::Check-Valid-Until no;' > /etc/apt/apt.conf.d/99no-check-valid-until
RUN apt-get update
RUN apt-get -y install make
# RUN apt-get -y upgrade swig
RUN apt-get -y install gcc
RUN apt-get -y install g++
RUN apt-get -y install build-essential

RUN conda create -n env python=3.6

WORKDIR /root
ADD src/ src/

# remove xgboost from the requirements, we'll install manually
RUN grep -v "xgboost" src/core/requirements.txt > src/core/requirements_no_xgboost.txt


RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate env && \
    cat src/core/requirements_no_xgboost.txt | xargs -n 1 -L 1 pip install -v


# build xgboost from source (use same version as pip installed usually
# but that won't work out of the box in all systems for some reason...)
# and need some fixes
ADD src/install_xgboost.sh .
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate env && \
    bash install_xgboost.sh

# just useful to have
RUN apt-get -y install vim
RUN apt-get -y install zip

RUN echo '#!/bin/bash \n\
. /opt/conda/etc/profile.d/conda.sh \n\
conda activate env \n\
' > /root/.bashrc

ENTRYPOINT ["bash"]
