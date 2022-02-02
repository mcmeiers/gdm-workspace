FROM ubuntu:latest
WORKDIR /app

# Install base python3
RUN \
    apt-get update        && \
    apt-get install --yes    \
        python3              \
        python3-pip          \
        build-essential      \
        gfortran             \
        liblapack-dev        \
        unzip                \
        wget              && \
    apt-get clean all

# set up software directory outside of /root for shifter
RUN mkdir -p /software/lib/python3/site-packages
ENV PIP_TARGET /software/lib/python3/site-packages
ENV PYTHONPATH /software/lib/python3/site-packages

# install various python packages
RUN python3 -m pip install --upgrade --no-cache-dir \
    matplotlib    \
    numpy         \
    cython        \
    scipy         \
    ipython

# install MPICH
ARG mpich=3.3
ARG mpich_prefix=mpich-$mpich

RUN \
    wget https://www.mpich.org/static/downloads/$mpich/$mpich_prefix.tar.gz && \
    tar xvzf $mpich_prefix.tar.gz                                           && \
    cd $mpich_prefix                                                        && \
    ./configure                                                             && \
    make -j 4                                                               && \
    make install                                                            && \
    make clean                                                              && \
    cd ..

# install mpi4py\
ARG mpi4py=3.0.3
ARG mpi4py_prefix=mpi4py-$mpi4py

RUN \
    wget https://bitbucket.org/mpi4py/mpi4py/downloads/$mpi4py_prefix.tar.gz && \
    tar xvzf $mpi4py_prefix.tar.gz                                           && \
    cd $mpi4py_prefix                                                        && \
    python3 setup.py build                                                   && \
    python3 setup.py install                                                 && \
    cd ..                                                                    && \
    rm -rf $mpi4py_prefix

# get cobaya
WORKDIR /software
RUN python3 -m pip install cobaya --upgrade --no-cache-dir

# install likelihood codes and data

RUN mkdir -p /software/cobaya_packages/code
RUN mkdir -p /software/cobaya_packages/data
ENV PYTHON=/usr/bin/python3

# install Planck data
WORKDIR /software/cobaya_packages/data
RUN wget -O COM_Likelihood_Data-baseline_R3.00.tar.gz http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_Likelihood_Data-baseline_R3.00.tar.gz
RUN tar xzvf COM_Likelihood_Data-baseline_R3.00.tar.gz
RUN rm COM_Likelihood_Data-baseline_R3.00.tar.gz


# install Planck likelihood code
RUN mkdir -p /software/cobaya_packages/code/planck
WORKDIR /software/cobaya_packages/code/planck
RUN wget -O COM_Likelihood_Code-v3.0_R3.10.tar.gz http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_Likelihood_Code-v3.0_R3.10.tar.gz
RUN tar xzvf COM_Likelihood_Code-v3.0_R3.10.tar.gz
RUN rm COM_Likelihood_Code-v3.0_R3.10.tar.gz
WORKDIR /software/cobaya_packages/code/planck/code/plc_3.0/plc-3.1
RUN PYTHON=/usr/bin/python3 ./waf configure --cfitsio_lib=/software/${cfits} --cfitsio_include=/software/${cfits} --pythondir=${PYTHONPATH}
RUN ./waf install

RUN mkdir -p /software/cobaya_packages/data/planck_2018
WORKDIR /software/cobaya_packages/data/planck_2018
RUN wget -O COM_PowerSpect_CMB-TT-full_R3.01.txt "http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_PowerSpect_CMB-TT-full_R3.01.txt"
RUN wget -O COM_PowerSpect_CMB-TE-full_R3.01.txt "http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_PowerSpect_CMB-TE-full_R3.01.txt"
RUN wget -O COM_PowerSpect_CMB-EE-full_R3.01.txt "http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_PowerSpect_CMB-EE-full_R3.01.txt"
RUN unzip

## install cobaya packages
#RUN cobaya-install planck --packages-path /software/cobaya_packages

## install CLASS
#WORKDIR /software/cobaya_packages/code
#RUN wget -O class_public.zip https://github.com/mcmeiers/class_public/archive/refs/heads/designer.zip
#RUN unzip class_public.zip
#RUN rm class_public.zip
#WORKDIR /software/cobaya_packages/code/class_public-designer/
#RUN PYTHON=python3 make -j 4


WORKDIR /app