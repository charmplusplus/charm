#ifndef _CONV_MACH_H
#define _CONV_MACH_H

/* define the default linker, together with its options */
#define CMK_DLL_CC   "g++ -shared -O3 -o "

/* 1 if the machine has a function called "getpagesize()", 0 otherwise .
   used in the memory files of converse */
#define CMK_GETPAGESIZE_AVAILABLE                          0

/* defines which version of memory handlers should be used.
   used in conv-core/machine.c */
#define CMK_MALLOC_USE_GNU_MALLOC                          0
#define CMK_MALLOC_USE_OS_BUILTIN                          1

#define CMK_MEMORY_PAGESIZE                                8192
#define CMK_MEMORY_PROTECTABLE                             1

/* the following definitions set the type of shared variables to be used. only
   one of them must be 1, all the others 0. The different implementations are in
   convserve.h Typically used are UNAVAILABLE for non SMP versions and
   POSIX_THREADS_SMP for SMP versions. The others are used only in special
   cases: UNIPROCESSOR in sim and uth, PTHREADS in origin,
   and NT_THREADS in windows. */
#define CMK_SHARED_VARS_UNAVAILABLE                        1 /* non SMP versions */
#define CMK_SHARED_VARS_POSIX_THREADS_SMP                  0 /* SMP versions */
#define CMK_SHARED_VARS_UNIPROCESSOR                       0
#define CMK_SHARED_VARS_NT_THREADS                         0

/* the following define if signal handlers should be used, both equal to zero
   means that signals will not be used. only one of the following can be 1, the
   other must be 0. they differ in the fact that the second (_WITH_RESTART)
   enables retry on interrupt (a function is recalled upon interrupt and does
   not return EINTR as in the first case) */
#define CMK_SIGNAL_USE_SIGACTION                           0
#define CMK_SIGNAL_USE_SIGACTION_WITH_RESTART              1

/* specifies whether the CthCpv variables should be defined as Cpv (0) or
   directly as normal c variables (1) */
#define CMK_THREADS_REQUIRE_NO_CPV                         0

/* decide which is the default implementation of the threads (see threads.c)
   Only one of the following can be 1. If none of them is selected, qthreads
   will be used as default. This default can be overwritten at compile time
   using -DCMK_THREADS_BUILD_"type"=1 */
#define CMK_THREADS_USE_CONTEXT                            0
#define CMK_THREADS_USE_JCONTEXT                           0
#define CMK_THREADS_USE_PTHREADS                           0

/* Specifies what kind of timer to use, and the correspondent headers will be
   included in convcore.c. If none is selected, then the machine.c file needs to
   implement the timer primitives. */
#define CMK_TIMER_USE_RTC                                  0
#define CMK_TIMER_USE_RDTSC                                0
#define CMK_TIMER_USE_GETRUSAGE                            0
#define CMK_TIMER_USE_SPECIAL                              0
#define CMK_TIMER_USE_TIMES                                0
#define CMK_TIMER_USE_BLUEGENEL                            0


/* In order to have a type with a fixed length across machines, these define the
   different size integers, unsigned integers, and floats as the machine
   specific types corresponding to the given sizes (2, 4, 8 bytes)

   Delete on systems where stdint.h is present and floating point
   formats are as defined in IEEE 754, respectively.
*/
#if defined(CMK_HAS_STDINT_H)
#error "Your system has stdint.h. Delete custom integer width definitions."
#else
#define CMK_TYPEDEF_INT2 short
#define CMK_TYPEDEF_INT4 int
#define CMK_TYPEDEF_INT8 long long
#define CMK_TYPEDEF_UINT2 unsigned short
#define CMK_TYPEDEF_UINT4 unsigned int
#define CMK_TYPEDEF_UINT8 unsigned long long
#endif

#define CMK_CUSTOM_FP_FORMAT
#define CMK_TYPEDEF_FLOAT4 something_like_float
#define CMK_TYPEDEF_FLOAT8 something_like_double

/* Specifies what the processor will do when it is idle, either sleep (1) or go
   into busy waiting mode (0). In convcore.c there are a few files included if
   sleeping mode, but the real distinct implementation is in the machine.c
   file. */
#define CMK_WHEN_PROCESSOR_IDLE_USLEEP                     1

/* specifies weather there is a web server collecting utilization statistics (1)
   or not (0) */
#define CMK_WEB_MODE                                       1

#define CMK_DEBUG_MODE                                     0

/* enables the load balancer framework. set to 1 for almost all the machines */
#define CMK_LBDB_ON					   1

/* snables smp support if set to 1 */
#define CMK_SMP                                            0

/* Other possible definitions:

In fault tolerant architectures, CK_MEM_CHECKPOINT can be set. In this case the
extended header must contain also another field called "pn" (phase number).

*/

#endif
