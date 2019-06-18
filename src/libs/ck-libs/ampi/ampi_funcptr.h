#ifndef AMPI_FUNCPTR_H_
#define AMPI_FUNCPTR_H_

#include "ampi.h"


#define AMPI_CUSTOM_FUNCPTR_DEF(return_type, function_name, ...) \
  return_type (* function_name)(__VA_ARGS__);
#if AMPI_HAVE_PMPI
  #define AMPI_FUNCPTR_DEF(return_type, function_name, ...) \
    return_type (* function_name)(__VA_ARGS__);             \
    return_type (* P##function_name)(__VA_ARGS__);
#else
  #define AMPI_FUNCPTR_DEF AMPI_CUSTOM_FUNCPTR_DEF
#endif


struct AMPI_FuncPtr_Transport
{
#define AMPI_FUNC AMPI_FUNCPTR_DEF
#define AMPI_FUNC_NOIMPL AMPI_FUNC
#define AMPI_CUSTOM_FUNC AMPI_CUSTOM_FUNCPTR_DEF

#include "ampi_functions.h"

#undef AMPI_FUNC
#undef AMPI_FUNC_NOIMPL
#undef AMPI_CUSTOM_FUNC
};


#endif /* AMPI_FUNCPTR_H_ */
