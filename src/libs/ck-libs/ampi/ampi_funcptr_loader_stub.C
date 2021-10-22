
#include "ampi_funcptr_loader.h"

void AMPI_FuncPtr_Pack(struct AMPI_FuncPtr_Transport * funcptrs)
{
  (void)funcptrs;
}

AMPI_FuncPtr_Unpack_t AMPI_FuncPtr_Unpack_Locate(SharedObject myexe)
{
  (void)myexe;
  return nullptr;
}
