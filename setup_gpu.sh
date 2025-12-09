if [ -d "./kokkos/install" ]; then
    echo "Kokkos Found at ${PWD}/kokkos/install"
else
    export KOKKOS_VERSION=4.7.01 # Replace with the actual version
    export KOKKOS_DOWNLOAD_URL=https://github.com/kokkos/kokkos/releases/download/${KOKKOS_VERSION}
    curl -sLO ${KOKKOS_DOWNLOAD_URL}/kokkos-${KOKKOS_VERSION}.tar.gz
    tar -xzvf kokkos-${KOKKOS_VERSION}.tar.gz
    rm kokkos-${KOKKOS_VERSION}.tar.gz
    mv kokkos-${KOKKOS_VERSION} kokkos
    cd kokkos
    rm -rf build
    mkdir build
    cd build

    # ensure that you have cmake/3.27.9 cuda/12.4.0 and eigen[for later] loaded

    ## for delta please run these before running the setup script
    # module load cuda/12.4.0
    # module load eigen
    # module load cmake/3.27.9

    ## The best practice is to let cmake autodetect the architecture, please run on a GPU syster or add a srun
    cmake -DBUILD_SHARED_LIBS=ON .. -DKokkos_ENABLE_CUDA=ON
    make -j${nproc}
    cd ..
    mkdir install
    cmake --install build --prefix install
    cd ..
fi

if [ -d "./kokkos-kernels/install" ]; then
    echo "Kokkos Kernels Found at ${PWD}/kokkos_kernels/install"
else
    git clone https://github.com/kokkos/kokkos-kernels.git
    export KOKKOS_KERNELS_VERSION=4.7.01 # Replace with the actual version
    export KOKKOS_DOWNLOAD_URL=https://github.com/kokkos/kokkos-kernels/releases/download/${KOKKOS_VERSION}
    curl -sLO ${KOKKOS_DOWNLOAD_URL}/kokkos-kernels-${KOKKOS_VERSION}.tar.gz
    tar -xzvf kokkos-kernels-${KOKKOS_VERSION}.tar.gz
    rm kokkos-kernels-${KOKKOS_VERSION}.tar.gz
    mv kokkos-kernels-${KOKKOS_VERSION} kokkos-kernels
    cd kokkos-kernels
    mkdir build
    cd build
    cmake .. -DCMAKE_CXX_COMPILER=g++ -DCMAKE_INSTALL_PREFIX=${PWD}/../install -DKokkos_ROOT=${PWD}/../../kokkos/install -DBUILD_SHARED_LIBS=ON -DCUBLAS_ROOT=/usr/local/cuda   -DCUSPARSE_ROOT=/usr/local/cuda -DCUSOLVER_ROOT=/usr/local/cuda
    make -j16
    cd .. 
    mkdir install
    cmake --install build --prefix install
    cd ..
fi
