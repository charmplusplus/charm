#!/bin/bash

./charmrun ./alltoall_VPtest +tcharm_stacksize 50000 +p 8 +vp 8 16
./charmrun ./alltoall_VPtest +tcharm_stacksize 50000 +p 8 +vp 16 16
./charmrun ./alltoall_VPtest +tcharm_stacksize 50000 +p 8 +vp 32 16
./charmrun ./alltoall_VPtest +tcharm_stacksize 50000 +p 8 +vp 64 16
./charmrun ./alltoall_VPtest +tcharm_stacksize 50000 +p 8 +vp 128 16
./charmrun ./alltoall_VPtest +tcharm_stacksize 50000 +p 8 +vp 256 16
./charmrun ./alltoall_VPtest +tcharm_stacksize 50000 +p 8 +vp 512 16
./charmrun ./alltoall_VPtest +tcharm_stacksize 50000 +p 8 +vp 1024 16
./charmrun ./alltoall_VPtest +tcharm_stacksize 50000 +p 8 +vp 2048 16

./charmrun ./alltoall_VPtest +tcharm_stacksize 50000 +p 8 +vp 8 1024
./charmrun ./alltoall_VPtest +tcharm_stacksize 50000 +p 8 +vp 16 1024
./charmrun ./alltoall_VPtest +tcharm_stacksize 50000 +p 8 +vp 32 1024
./charmrun ./alltoall_VPtest +tcharm_stacksize 50000 +p 8 +vp 64 1024
./charmrun ./alltoall_VPtest +tcharm_stacksize 50000 +p 8 +vp 128 1024
./charmrun ./alltoall_VPtest +tcharm_stacksize 50000 +p 8 +vp 256 1024
./charmrun ./alltoall_VPtest +tcharm_stacksize 50000 +p 8 +vp 512 1024
./charmrun ./alltoall_VPtest +tcharm_stacksize 50000 +p 8 +vp 1024 1024

./charmrun ./alltoall_VPtest +tcharm_stacksize 50000 +p 8 +vp 8 16000
./charmrun ./alltoall_VPtest +tcharm_stacksize 50000 +p 8 +vp 16 16000
./charmrun ./alltoall_VPtest +tcharm_stacksize 50000 +p 8 +vp 32 16000
./charmrun ./alltoall_VPtest +tcharm_stacksize 50000 +p 8 +vp 64 16000
./charmrun ./alltoall_VPtest +tcharm_stacksize 50000 +p 8 +vp 128 16000
./charmrun ./alltoall_VPtest +tcharm_stacksize 50000 +p 8 +vp 256 16000


./charmrun ./alltoall_VPtest +tcharm_stacksize 50000 +p 8 +vp 8 1048576
./charmrun ./alltoall_VPtest +tcharm_stacksize 50000 +p 8 +vp 16 1048576
./charmrun ./alltoall_VPtest +tcharm_stacksize 50000 +p 8 +vp 32 1048576
./charmrun ./alltoall_VPtest +tcharm_stacksize 50000 +p 8 +vp 64 1048576
./charmrun ./alltoall_VPtest +tcharm_stacksize 50000 +p 8 +vp 128 1048576
./charmrun ./alltoall_VPtest +tcharm_stacksize 50000 +p 8 +vp 256 1048576 
