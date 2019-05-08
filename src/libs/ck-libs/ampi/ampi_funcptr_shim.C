// This object can be linked to AMPI binaries in place of the RTS.

#ifndef AMPI_USE_FUNCPTR
# error This file requires -ampi-funcptr-shim.
#endif
#include "ampi_funcptr.h"


// Provide the definitions of function pointers corresponding to the entire AMPI API.

#define AMPI_FUNC AMPI_FUNCPTR_DEF
#define AMPI_FUNC_NOIMPL AMPI_FUNC
#define AMPI_CUSTOM_FUNC AMPI_CUSTOM_FUNCPTR_DEF

#include "ampi_functions.h"

#undef AMPI_FUNC
#undef AMPI_FUNC_NOIMPL
#undef AMPI_CUSTOM_FUNC


// Provide an interface to link the function pointers at runtime.

extern "C" void AMPI_FuncPtr_Unpack(struct AMPI_FuncPtr_Transport * x)
{
#define AMPI_CUSTOM_FUNC(return_type, function_name, ...) \
  function_name = x->function_name;
#if AMPI_HAVE_PMPI
  #define AMPI_FUNC(return_type, function_name, ...) \
    function_name = x->function_name; \
    P##function_name = x->P##function_name;
#else
  #define AMPI_FUNC AMPI_CUSTOM_FUNC
#endif
#define AMPI_FUNC_NOIMPL AMPI_FUNC

#include "ampi_functions.h"

#undef AMPI_FUNC
#undef AMPI_FUNC_NOIMPL
#undef AMPI_CUSTOM_FUNC
}


// Provide a stub entry point so the program will link without any special effort.

#include <stdio.h>

#ifdef main
# undef main
#endif
int main()
{
  fprintf(stderr, "Do not run this binary directly!\n");
  return 1;
}
