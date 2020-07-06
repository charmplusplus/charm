
#include "ampi_funcptr_loader.h"

#include <stdio.h>
#include <string.h>


static void AMPI_FuncPtr_Pack(struct AMPI_FuncPtr_Transport * x)
{
#define AMPI_CUSTOM_FUNC(return_type, function_name, ...) \
    x->function_name = function_name;
#if AMPI_HAVE_PMPI
  #define AMPI_FUNC(return_type, function_name, ...) \
      x->function_name = function_name; \
      x->P##function_name = P##function_name;
#else
  #define AMPI_FUNC AMPI_CUSTOM_FUNC
#endif
#define AMPI_FUNC_NOIMPL AMPI_FUNC

#include "ampi_functions.h"

#undef AMPI_FUNC
#undef AMPI_FUNC_NOIMPL
#undef AMPI_CUSTOM_FUNC
}

static void AMPI_FuncPtr_Unpack_Dispatch(SharedObject myexe, struct AMPI_FuncPtr_Transport * x)
{
  typedef int (*myPtrUnpackType)(struct AMPI_FuncPtr_Transport *);
  auto myPtrUnpack = (myPtrUnpackType)dlsym(myexe, "AMPI_FuncPtr_Unpack");

  if (myPtrUnpack == nullptr)
  {
    CkError("dlsym error: %s\n", dlerror());
    CkAbort("Could not complete AMPI_FuncPtr_Unpack!");
  }

  myPtrUnpack(x);
}


int AMPI_FuncPtr_Loader(SharedObject myexe, int argc, char ** argv)
{
  // populate the user binary's function pointer shim
  {
    AMPI_FuncPtr_Transport x;
    AMPI_FuncPtr_Pack(&x);
    AMPI_FuncPtr_Unpack_Dispatch(myexe, &x);
  }

  // jump to the user binary
  return AMPI_Main_Dispatch(myexe, argc, argv);
}
