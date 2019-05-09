#ifndef AMPI_FUNCPTR_LOADER_H_
#define AMPI_FUNCPTR_LOADER_H_

#ifndef __STDC_FORMAT_MACROS
# define __STDC_FORMAT_MACROS
#endif
#ifndef __STDC_LIMIT_MACROS
# define __STDC_LIMIT_MACROS
#endif
#include <inttypes.h>
#include <limits.h>

#include "ampiimpl.h"
#include "ampi_funcptr.h"

#define STRINGIZE_INTERNAL(x) #x
#define STRINGIZE(x) STRINGIZE_INTERNAL(x)


#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
# define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
# define NOMINMAX
#endif
#include <windows.h>

typedef HMODULE SharedObject;

#define dlopen(name, flags) LoadLibrary(name)
#define dlsym(handle, name) GetProcAddress((handle), (name))
#define dlclose(handle) FreeLibrary(handle)

#else

#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif
#ifndef __USE_GNU
# define __USE_GNU
#endif
#include <dlfcn.h>

typedef void * SharedObject;

#endif


int AMPI_FuncPtr_Loader(SharedObject, int, char **);

#endif /* AMPI_FUNCPTR_LOADER_H_ */
