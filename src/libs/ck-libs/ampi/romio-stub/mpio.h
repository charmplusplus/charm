/* dummy mpio.h for inclusion in ampi.h when ROMIO is not built */

#ifndef MPIO_H_DUMMY
#define MPIO_H_DUMMY

// Provide this type for not implemented functions in case ROMIO is not built,
// or when building AMPI itself.
typedef void* MPI_File;

#endif
