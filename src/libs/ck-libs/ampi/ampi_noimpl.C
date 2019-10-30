#include "ampiimpl.h"

/*
This file contains function definitions of all MPI functions that are currently
unsupported in AMPI. Calling these functions aborts the application.
*/

#define AMPI_NOIMPL_ONLY

#define AMPI_NOIMPL_BODY(function_name) \
    { \
        AMPI_API(STRINGIFY(function_name), ""); \
        CkAbort(STRINGIFY(function_name) " is not implemented in AMPI."); \
    }

// This is its own macro because otherwise we will get PAMPI instead of APMPI.
// keep in sync with ampiimpl.h
#if AMPI_HAVE_PMPI
  #define AMPI_FUNC_NOIMPL(ret, name, ...) \
    CLINKAGE \
    __attribute__((weak, alias(STRINGIFY(name)))) \
    ret P##name(__VA_ARGS__); \
    CLINKAGE \
    __attribute__((weak)) \
    ret name(__VA_ARGS__) \
    AMPI_NOIMPL_BODY(name)
#else
  #define AMPI_FUNC_NOIMPL(ret, name, ...) \
    CLINKAGE \
    ret name(__VA_ARGS__) \
    AMPI_NOIMPL_BODY(name)
#endif

#include "ampi_functions.h"

#undef AMPI_NOIMPL_ONLY
#undef AMPI_FUNC_NOIMPL
