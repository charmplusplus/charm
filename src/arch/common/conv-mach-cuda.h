#undef  CMK_CUDA
#define CMK_CUDA                                           1

/* Either vendor: the GPU runtime (hybridAPI) is built in. Defined here as
   well as in conv-config.h so that a file including only conv-mach-opt.h
   (an interface file run through charmxi) sees it. */
#undef  CMK_GPU
#define CMK_GPU                                            1

#undef CMK_WHEN_PROCESSOR_IDLE_BUSYWAIT
#define CMK_WHEN_PROCESSOR_IDLE_BUSYWAIT                   1
#undef CMK_WHEN_PROCESSOR_IDLE_USLEEP
#define CMK_WHEN_PROCESSOR_IDLE_USLEEP                     0
