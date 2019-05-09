/*
 * This file is separate from hapi.h because the Hybrid API is included as part
 * of AMPI's extensions to the MPI standard, and certain global variable
 * privatization methods require the AMPI API to be exposed as function pointers
 * through a shim and loader mechanism that needs to list the entire set of
 * provided functions at multiple points in its implementation.
 *
 * See src/libs/ck-libs/ampi/ampi_functions.h for mandatory procedures.
 *
 * For ease of reading: AMPI_CUSTOM_FUNC(ReturnType, FunctionName, Parameters...)
 */

/******************** DEPRECATED ********************/
// Create a hapiWorkRequest object for the user. The runtime manages the associated
// memory, so the user only needs to set it up properly.
AMPI_CUSTOM_FUNC(hapiWorkRequest*, hapiCreateWorkRequest, void)

/******************** DEPRECATED ********************/
// Add a work request into the "queue". Currently all specified data transfers
// and kernel execution are directly put into a CUDA stream.
AMPI_CUSTOM_FUNC(void, hapiEnqueue, hapiWorkRequest* wr)

// The runtime queries the compute capability of the device, and creates as
// many streams as the maximum number of concurrent kernels.
AMPI_CUSTOM_FUNC(int, hapiCreateStreams, void)

// Get a CUDA stream that was created by the runtime. Current scheme is to
// hand out streams in a round-robin fashion.
AMPI_CUSTOM_FUNC(cudaStream_t, hapiGetStream, void)

// Add a Charm++ callback function to be invoked after the previous operation
// in the stream completes. This call should be placed after data transfers or
// a kernel invocation.
AMPI_CUSTOM_FUNC(void, hapiAddCallback, cudaStream_t, void*, void*)

// Thin wrappers for memory related CUDA API calls.
AMPI_CUSTOM_FUNC(cudaError_t, hapiMalloc, void**, size_t)
AMPI_CUSTOM_FUNC(cudaError_t, hapiFree, void*)
AMPI_CUSTOM_FUNC(cudaError_t, hapiMallocHost, void**, size_t)
AMPI_CUSTOM_FUNC(cudaError_t, hapiFreeHost, void*)
AMPI_CUSTOM_FUNC(cudaError_t, hapiMallocHostPool, void**, size_t)
AMPI_CUSTOM_FUNC(cudaError_t, hapiFreeHostPool, void*)
AMPI_CUSTOM_FUNC(cudaError_t, hapiMemcpyAsync, void*, const void*, size_t, enum cudaMemcpyKind, cudaStream_t)

// Explicit memory allocations using pinned memory pool.
AMPI_CUSTOM_FUNC(void*, hapiPoolMalloc, size_t)
AMPI_CUSTOM_FUNC(void, hapiPoolFree, void*)

// Provides support for detecting errors with CUDA API calls.
AMPI_CUSTOM_FUNC(void, hapiErrorDie, cudaError_t, const char*, const char*, int)

#ifdef HAPI_INSTRUMENT_WRS
AMPI_CUSTOM_FUNC(void, hapiInitInstrument, int n_chares, char n_types)
AMPI_CUSTOM_FUNC(hapiRequestTimeInfo*, hapiQueryInstrument, int chare, char type, char phase)
#endif
