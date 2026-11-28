#undef  CMK_CUDA
#define CMK_CUDA                                           1

/* The GPU runtime (hybridAPI) is built in. Runtime code that does not depend
   on the vendor tests CMK_GPU; only vendor-specific code tests CMK_CUDA.
   Defined here so that a file including only conv-mach-opt.h (an interface
   file run through charmxi) sees it. */
#undef  CMK_GPU
#define CMK_GPU                                            1

#undef CMK_WHEN_PROCESSOR_IDLE_BUSYWAIT
#define CMK_WHEN_PROCESSOR_IDLE_BUSYWAIT                   1
#undef CMK_WHEN_PROCESSOR_IDLE_USLEEP
#define CMK_WHEN_PROCESSOR_IDLE_USLEEP                     0
