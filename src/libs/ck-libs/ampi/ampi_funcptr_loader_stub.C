
#include "ampi_funcptr_loader.h"

int AMPI_FuncPtr_Loader(SharedObject myexe, int argc, char ** argv)
{
  // jump to the user binary
  return AMPI_Main_Dispatch(myexe, argc, argv);
}
