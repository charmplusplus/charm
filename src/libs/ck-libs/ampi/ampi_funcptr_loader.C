// This file is linked with AMPI globals launching programs to facilitate symbol namespacing.

#ifdef AMPI_USE_FUNCPTR
# error This file must *not* be built with -DAMPI_USE_FUNCPTR.
#endif

#include "ampi_funcptr_loader.h"

#include <stdio.h>
#include <string.h>

void AMPI_FuncPtr_Pack(struct AMPI_FuncPtr_Transport * funcptrs)
{
#define AMPI_CUSTOM_FUNC(return_type, function_name, ...) \
    funcptrs->function_name = &function_name;
#if AMPI_HAVE_PMPI
  #define AMPI_FUNC(return_type, function_name, ...) \
      funcptrs->function_name = &function_name; \
      funcptrs->P##function_name = &P##function_name;
#else
  #define AMPI_FUNC AMPI_CUSTOM_FUNC
#endif
#define AMPI_FUNC_NOIMPL AMPI_FUNC

#include "ampi_functions.h"

#undef AMPI_FUNC
#undef AMPI_FUNC_NOIMPL
#undef AMPI_CUSTOM_FUNC
}

AMPI_FuncPtr_Unpack_t AMPI_FuncPtr_Unpack_Locate(SharedObject myexe)
{
  auto myPtrUnpack = (AMPI_FuncPtr_Unpack_t)dlsym(myexe, "AMPI_FuncPtr_Unpack");

  if (myPtrUnpack == nullptr)
  {
    CkError("dlsym error: %s\n", dlerror());
    CkAbort("Could not complete AMPI_FuncPtr_Unpack!");
  }

  return myPtrUnpack;
}
