This benchmark utilizes busywaiting to simulate work both on the CPU and GPU.
It was designed to test the performance of the new Hybrid API changes.
A custom map is used to map GPU chares to GPU handler PEs, and the CPU chares
to the normal PEs.

Example: ./busywait +p5 -c 8 -a 0.1 -r 0.5 -y -g 1
         This command will run the benchmark with 4 normal PEs and 1 GPU PE,
         where the GPU PE houses all chares that run only on the GPU. Since
         the offload percentage is set to 50%, 4 chares will run on the 4
         normal PEs, and the other 4 chares will run on the GPU PE. The
         busywait time will be 100 ms.
