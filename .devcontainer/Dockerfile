FROM docker.io/continuumio/miniconda3:4.10.3

# Install base python3
RUN \
    apt-get update        && \
    apt-get install --yes    \
        build-essential      \
        gfortran             \
        liblapack-dev        \
        libopenblas-dev      \
        zlib1g-dev           \
        unzip                \
        wget              && \
    apt-get clean all

ENV PATH=/opt/conda/bin:$PATH

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



# set up software directory outside of /root for shifter
ENV PYTHONPATH /opt/conda/lib/python3.9/site-packages

# install various python packages
RUN /opt/conda/bin/conda install -c conda-forge \
    matplotlib    \
    numpy         \
    cython        \
    scipy         \
    ipython       \
    ruamel.yaml   \
    ipykernel     \
    mpi4py        

RUN python -m pip install cobaya==3.2.1 --upgrade
RUN python -m pip install getdist

# install cfits
WORKDIR /software
ARG cfits=cfitsio-4.0.0
RUN wget -O ${cfits}.tar.gz http://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/${cfits}.tar.gz
RUN tar xvfz ${cfits}.tar.gz
RUN rm ${cfits}.tar.gz
WORKDIR /software/${cfits}
RUN ./configure
RUN make shared


# install cobaya pacakges
RUN python -m cobaya install -p /software/cobaya_packages planck_2018_lowl.TT planck_2018_lowl.EE 

# install CLASS
WORKDIR /software/cobaya_packages/code
RUN wget -O class_public.zip https://github.com/mcmeiers/class_public/archive/refs/heads/designer.zip
RUN unzip class_public.zip
RUN rm class_public.zip
WORKDIR /software/cobaya_packages/code/class_public-designer/
RUN make -j 4 PYTHON=python


WORKDIR /opt/project
RUN mkdir output

