/* for ChaMPIon/Pro 1.1.1 (c) 1997 - 2004 MPI Software Technology, Inc. */
/* on Tungsten.ncsa */

#undef CMK_MALLOC_USE_GNU_MALLOC
#undef CMK_MALLOC_USE_OS_BUILTIN
#define CMK_MALLOC_USE_GNU_MALLOC                          0
#define CMK_MALLOC_USE_OS_BUILTIN                          1

/* keep using QT, context thread migration is broken (megampi) on tungsten */
/* however QT crash megatest marshall test */
/* the reason cmpirun/gm crashed megatest seems to be some problem related
   to GM message protocol for long messages, using -gm_long <size> seems to 
   work around it, so now switch it back to QT */
#if 0
#undef CMK_THREADS_USE_CONTEXT
#define CMK_THREADS_USE_CONTEXT                            1
#endif

#undef CMK_TYPEDEF_INT8
#undef CMK_TYPEDEF_UINT8
#define CMK_TYPEDEF_INT8 long long
#define CMK_TYPEDEF_UINT8 unsigned long long
