
#define CMK_SMP						   1


/* Win32 version always uses shared variables for now.
#undef CMK_SHARED_VARS_UNAVAILABLE
#undef CMK_SHARED_VARS_NT_THREADS
#define CMK_SHARED_VARS_UNAVAILABLE                        0
#define CMK_SHARED_VARS_NT_THREADS                         1
*/

#undef CMK_THREADS_USE_JCONTEXT
#undef CMK_THREADS_ARE_WIN32_FIBERS
#define CMK_THREADS_USE_JCONTEXT                           0
#define CMK_THREADS_ARE_WIN32_FIBERS                       1


#define CMK_PCQUEUE_LOCK                                   1
