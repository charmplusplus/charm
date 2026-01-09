make clean
make verify
./charmrun ++local ++p 2 ./verify +gpushm +gpuipceventpool 512 +allgpus +gpucommbuffer 128
