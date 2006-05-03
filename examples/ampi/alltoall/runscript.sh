#!/bin/bash

./charmrun ./alltoall_VPtest +tcharm_stacksize 100000 +p 2 +vp 8 16
./charmrun ./alltoall_VPtest +tcharm_stacksize 100000 +p 2 +vp 16 16
./charmrun ./alltoall_VPtest +tcharm_stacksize 100000 +p 2 +vp 32 16
./charmrun ./alltoall_VPtest +tcharm_stacksize 100000 +p 2 +vp 64 16
./charmrun ./alltoall_VPtest +tcharm_stacksize 100000 +p 2 +vp 128 16
./charmrun ./alltoall_VPtest +tcharm_stacksize 100000 +p 2 +vp 256 16
./charmrun ./alltoall_VPtest +tcharm_stacksize 100000 +p 2 +vp 512 16
./charmrun ./alltoall_VPtest +tcharm_stacksize 100000 +p 2 +vp 1024 16

./charmrun ./alltoall_VPtest +tcharm_stacksize 100000 +p 2 +vp 8 1024
./charmrun ./alltoall_VPtest +tcharm_stacksize 100000 +p 2 +vp 16 1024
./charmrun ./alltoall_VPtest +tcharm_stacksize 100000 +p 2 +vp 32 1024
./charmrun ./alltoall_VPtest +tcharm_stacksize 100000 +p 2 +vp 64 1024
./charmrun ./alltoall_VPtest +tcharm_stacksize 100000 +p 2 +vp 128 1024
./charmrun ./alltoall_VPtest +tcharm_stacksize 100000 +p 2 +vp 256 1024

./charmrun ./alltoall_VPtest +tcharm_stacksize 100000 +p 2 +vp 8 16000
./charmrun ./alltoall_VPtest +tcharm_stacksize 100000 +p 2 +vp 16 16000
./charmrun ./alltoall_VPtest +tcharm_stacksize 100000 +p 2 +vp 32 16000
./charmrun ./alltoall_VPtest +tcharm_stacksize 100000 +p 2 +vp 64 16000
./charmrun ./alltoall_VPtest +tcharm_stacksize 100000 +p 2 +vp 128 16000

./charmrun ./alltoall_VPtest +tcharm_stacksize 100000 +p 2 +vp 8 1000000
./charmrun ./alltoall_VPtest +tcharm_stacksize 100000 +p 2 +vp 16 1000000
./charmrun ./alltoall_VPtest +tcharm_stacksize 100000 +p 2 +vp 32 1000000
