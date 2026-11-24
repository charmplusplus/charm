This example passes a Hello message along the elements of a chare array.
Each chare launches a kernel on the GPU when it receives the message. When the
kernel completes, the runtime system executes the specified callback function
which passes the message to the subsequent chare in the array.

The kernel fills a small buffer with a value derived from the chare index, and
the callback checks it. That check is the point: an empty kernel would make a
launch that silently never ran -- for instance because the object was built
without code for this device's architecture -- indistinguishable from a
successful one. Set CUDA_ARCH in the Makefile to match your GPU.

Usage: ./hello -c [chares]
