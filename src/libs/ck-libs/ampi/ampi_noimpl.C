#include "ampiimpl.h"

/*
This file contains function definitions of all MPI functions that are currently
unsupported in AMPI. Calling these functions aborts the application.
*/

#define AMPI_NOIMPL_ONLY
#define AMPI_FUNC_NOIMPL(return_type, function_name, ...) \
    AMPI_API_IMPL(return_type, function_name, __VA_ARGS__) \
    { \
        AMPI_API(STRINGIFY(function_name)); \
        CkAbort(STRINGIFY(function_name) " is not implemented in AMPI."); \
    }

#include "ampi_functions.h"

#undef AMPI_NOIMPL_ONLY
#undef AMPI_FUNC_NOIMPL
