This example performs 2D 5-point stencil operations for a given number of
iterations with a 2D chare array. Each chare in the array computes a portion
of the global temperature grid, and communicates only the ghost (halo) regions
with its neighbors. The GPU code has been optimized to transfer only the ghost
regions to/from the device, retaining the internal data on the GPU, aside from
the intial and final transfers that require transfer of the entire temperature
block of the chare.

The default execution mode is pure Charm++; specifing -u or -g as a command
line option will set the execution mode to either Charm++ with CUDA or Charm++
with Hybrid API, respectively, where the GPU is used for local stencil
computation. The offload ratio (specified as a float value between 0 and 1)
determines what percentage of chares are offloaded to the GPU. If it is set to
0, all chares will use the CPU (effectively the same as not specifying -u or
-g), if it is set to 1, all chares will use the GPU, and if it is set to 0.5,
half of the chares will be offloaded to the GPU for stencil computation.

The number of GPU threads utilized per chare may be specified, in which case
each thread will compute more than one element. This coarsening implementation
has not been optimized and could perform worse than when running without it.
However, it could be used to showcase the performance improvement provided by
HAPI, as the coarsening makes it possible to set the number of threads so that
a single chare's kernel does not occupy the entire GPU device. This allows
multiple kernels to execute concurrently on the device along with other chares
doing stencil on the CPU, which increases the effectiveness of the overlap of
heterogeneous tasks provided by HAPI.

Usage: ./stencil2d -s [grid size] -b [block size] -i [iterations]
                   -u/y: CUDA/HAPI -r [offload ratio]
                   -t [thread coarsening size]

Example: ./stencil2d -s 4096 -b 256 -i 100 -y -r 0.5 -t 4
         This will run with 16 x 16 = 256 chares, with 128 chares performing
         stencil on the CPU and the other 128 chares on the GPU. Each thread
         will calculate 4 x 4 = 16 elements.
