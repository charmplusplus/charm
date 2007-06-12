#ifndef _CONV_MACH_H
#define _CONV_MACH_H

#define CMK_IA64                                           1

#define CMK_MEMORY_PAGESIZE                                16384

#define CMK_THREADS_USE_CONTEXT                            1
#define CMK_THREADS_COPY_STACK                             0
#define CMK_THREADS_USE_PTHREADS                           0
#define CMK_THREADS_ARE_WIN32_FIBERS                       0

#define CMK_THREADS_REQUIRE_NO_CPV                         0

/*
 * The following three #define's are needed for running TCHARM/AMPI
 * codes in Grid environments.
 *
 * When a TCHARM/AMPI computation starts up, Node 0 probes for two
 * pieces of information: function pointers needed for entry into
 * the TCharm thread startup function and the AMPI main() function,
 * and the isomalloc memory map for thread stacks.
 *
 * The problem with probing the function pointers is that these
 * can be different on heterogeneous nodes in a Grid environment.
 * A simple hack is to set them statically, which the #define's do.
 *
 * The problem with probing the isomalloc memory map for thread stacks
 * is that valid locations can be different on heterogeneous nodes in a
 * Grid environment.  A simple hack is to disable isomalloc, which
 * means that threads cannot be migrated (must run with +tcharm_nomig).
*/
#define CMK_NO_ISO_MALLOC                                  0
#define CMK_TCHARM_FNPTR_HACK                              0
#define CMK_AMPI_FNPTR_HACK                                0

#endif
