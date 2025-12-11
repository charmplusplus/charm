rm -rf build
mkdir build
cd build
cmake -DCharm_ENABLE_GPU=ON ..
make -j16
./charmrun ++local ./jacobi2d.out -y -z

