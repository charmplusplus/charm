#if !defined(CMITLS_H)
#define CMITLS_H

#include "conv-config.h"

#if CMK_HAS_ELF_H && CMK_HAS_TLS_VARIABLES

#include <elf.h>
#include <string.h>
#include <stdlib.h>
#if CMK_HAS_MALLOC_H
#include <malloc.h>
#endif

#if ( defined(__LP64__) || defined(_LP64) )
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

typedef struct tlsseg {
  Addr memseg;
  size_t size;
  size_t align;
} tlsseg_t;


#else

typedef int  tlsseg_t;            /* place holder */

#endif

#endif
