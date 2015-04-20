#define CMK_SMP						   1


#undef CMK_SHARED_VARS_UNAVAILABLE
#undef CMK_SHARED_VARS_POSIX_THREADS_SMP
#define CMK_SHARED_VARS_UNAVAILABLE                        0
#define CMK_SHARED_VARS_POSIX_THREADS_SMP                  1

#undef CMK_THREADS_USE_JCONTEXT
#define CMK_THREADS_USE_JCONTEXT                           1

#undef CMK_USE_POLL
#define CMK_USE_POLL                                       1
#undef CMK_USE_KQUEUE
#define CMK_USE_KQUEUE                                     0

/*#if !CMK_GCC_X86_ASM || !CMK_GCC_X86_ASM_ATOMICINCREMENT */
/* #define CMK_PCQUEUE_LOCK                                    1 */
/*#endif */
