#ifndef _CONV_MACH_PXSHM_
#define _CONV_MACH_PXSHM_

#undef CMK_USE_PXSHM
#define CMK_USE_PXSHM 			1

#undef CMK_IMMEDIATE_MSG
#define CMK_IMMEDIATE_MSG       1

#undef CMK_WHEN_PROCESSOR_IDLE_USLEEP
#define CMK_WHEN_PROCESSOR_IDLE_USLEEP  0

#define PXSHM_LOCK                      1

#endif
