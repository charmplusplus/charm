#undef  CMK_CUDA
#define CMK_CUDA                                           1

#undef CMK_WHEN_PROCESSOR_IDLE_BUSYWAIT
#define CMK_WHEN_PROCESSOR_IDLE_BUSYWAIT                   1
#undef CMK_WHEN_PROCESSOR_IDLE_USLEEP
#define CMK_WHEN_PROCESSOR_IDLE_USLEEP                     0

#undef CMK___int128_DEFINED
#undef CMK___int128_t_DEFINED
