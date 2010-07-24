
#include "conv-config.h"

#if CMK_HAS_ELF_H && CMK_TLS_THREAD

#include <elf.h>
#include <string.h>
#include <stdlib.h>
#include <malloc.h>

#if ( __LP64__ || _LP64 )
#define ELF64
#else
#define ELF32
#endif

#ifdef ELF32
typedef Elf32_Addr Addr;
typedef Elf32_Ehdr Ehdr;
typedef Elf32_Phdr Phdr;
#else
typedef Elf64_Addr Addr;
typedef Elf64_Ehdr Ehdr;
typedef Elf64_Phdr Phdr;
#endif

typedef struct {
  Addr memseg;
  size_t size;
  size_t align;
} tlsseg_t;


#endif
