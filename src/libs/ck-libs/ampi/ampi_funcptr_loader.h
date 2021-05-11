#ifndef AMPI_FUNCPTR_LOADER_H_
#define AMPI_FUNCPTR_LOADER_H_

#include "ampiimpl.h"
#include "ampi_funcptr.h"

void AMPI_FuncPtr_Pack(struct AMPI_FuncPtr_Transport *);

typedef int (*AMPI_FuncPtr_Unpack_t)(const struct AMPI_FuncPtr_Transport *);
AMPI_FuncPtr_Unpack_t AMPI_FuncPtr_Unpack_Locate(SharedObject);

#endif /* AMPI_FUNCPTR_LOADER_H_ */
