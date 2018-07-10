
#include <stdio.h>

#include "converse.h"
#include "cmitls.h"

#if CMK_HAS_ELF_H && CMK_HAS_TLS_VARIABLES && CMK_DLL_USE_DLOPEN && CMK_HAS_RTLD_DEFAULT

/* These macros are needed for:
 * dlfcn.h: RTLD_DEFAULT
 */
#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif
#ifndef __USE_GNU
# define __USE_GNU
#endif
#include "dlfcn.h"

void* getTLS(void) CMI_NOOPTIMIZE;
void setTLS(void* newptr) CMI_NOOPTIMIZE;
void* swapTLS(void* newptr) CMI_NOOPTIMIZE;

CMI_EXTERNC_VARIABLE int quietModeRequested;

static void* CmiTLSExecutableStart;

void CmiTLSInit(void)
{
  if (CmiMyRank() == 0)
  {
    if (!quietModeRequested && CmiMyPe() == 0)
      CmiPrintf("Charm++> -tlsglobals enabled for privatization of thread-local variables.\n");

    /* Use dynamic linking in case Charm++ shared objects are used by a binary lacking
     * conv-static.o, such as in the case of CharmPy. */
    void** pCmiExecutableStart = (void**)dlsym(RTLD_DEFAULT, "CmiExecutableStart");
    if (pCmiExecutableStart != NULL)
      CmiTLSExecutableStart = *pCmiExecutableStart;
    else
      CmiPrintf("Charm++> Error: \"CmiExecutableStart\" symbol not found. -tlsglobals disabled.\n");
  }
}

static Addr getCodeSegAddr(void) {
  return (Addr) CmiTLSExecutableStart;
}

static Ehdr* getELFHeader(void) {
  return (Ehdr*) getCodeSegAddr();
}

static Phdr* getProgramHeader(Ehdr* ehdr) {
  return (Phdr*)((char *)ehdr + ehdr->e_phoff);
}

Phdr* getTLSPhdrEntry(void) {
  int phnum, i;
  Ehdr* elfHeader;
  Phdr* progHeader;

  elfHeader = getELFHeader();
  if (elfHeader == NULL)
    return NULL;

  phnum = elfHeader->e_phnum;
  progHeader = getProgramHeader(elfHeader);
  for (i = 0; i < phnum; i++) {
    if (progHeader[i].p_type == PT_TLS) {
#if CMK_ERROR_CHECKING
      /* sanity check */
      /* align is power of 2 */
      int align = progHeader[i].p_align;
      CmiAssert(align > 0 && ( (align & (align-1)) == 0));
      /* total size is not less than the size of .tdata (initializer data) */
      CmiAssert(progHeader[i].p_memsz >= progHeader[i].p_filesz);
#endif
      return &progHeader[i];
    }
  }
  return NULL;
}

void allocNewTLSSeg(tlsseg_t* t, CthThread th) {
  Phdr* phdr;

  phdr = getTLSPhdrEntry();
  if (phdr != NULL) {
    t->align = phdr->p_align;
    t->size = CMIALIGN(phdr->p_memsz, phdr->p_align);
    t->memseg = (Addr)CmiIsomallocAlign(t->align, t->size, th);
    memset((void*)t->memseg, 0, t->size);
    memcpy((void*)t->memseg, (void*) (phdr->p_vaddr), (size_t)(phdr->p_filesz));
    t->memseg = (Addr)( ((char *)(t->memseg)) + t->size );
    /* printf("[%d] 2 ALIGN %d MEM %p SIZE %d\n", CmiMyPe(), t->align, t->memseg, t->size); */
  } else {
    /* since we don't have a PT_TLS section to copy, keep whatever the system gave us */
    t->memseg = (Addr)getTLS();
    t->size = 0;
    t->align = 0;
  }
}

void switchTLS(tlsseg_t* , tlsseg_t* ) CMI_NOOPTIMIZE;
void currentTLS(tlsseg_t*) CMI_NOOPTIMIZE;

/* void allocNewTLSSegEmpty(tlsseg_t* t) {
 *   void* aux = CmiIsomallocAlign(t->align, t->size);
 *     t->memseg = (Addr) (aux + t->size);
 *     } */

void currentTLS(tlsseg_t* cur) {
  cur->memseg = (Addr)getTLS();
}

void* getTLS(void) {
  void* ptr;
#if CMK_TLS_SWITCHING64
  asm volatile ("movq %%fs:0x0, %0\n\t"
                : "=r"(ptr));
#elif CMK_TLS_SWITCHING32
  asm volatile ("movl %%gs:0x0, %0\n\t"
                : "=r"(ptr));
#else
  fprintf(stderr, "TLS globals are not supported.");
  abort();
#endif
  return ptr;
}

void switchTLS(tlsseg_t* cur, tlsseg_t* next) {
  cur->memseg = (Addr)swapTLS((void*)next->memseg);
}

void* swapTLS(void* newptr) {
  void* oldptr;
#if CMK_TLS_SWITCHING64
#if 0
  asm volatile ("movq %%fs:0x0, %0\n\t"
                : "=r"(oldptr));
  if (oldptr == newptr) /* same */
    return oldptr;
  asm volatile ("movq %0, %%fs:0x0\n\t"
                :
                : "r"(newptr));
#else
  asm volatile ("movq %%fs:0x0, %0\n\t"
                "movq %1, %%fs:0x0\n\t"
                : "=&r"(oldptr)
                : "r"(newptr));
#endif
#elif CMK_TLS_SWITCHING32
  asm volatile ("movl %%gs:0x0, %0\n\t"
                "movl %1, %%gs:0x0\n\t"
                : "=&r"(oldptr)
                : "r"(newptr));
#else
  fprintf(stderr, "TLS globals are not supported.");
  abort();
#endif
  return oldptr;
}

/* for calling from a debugger */
void setTLS(void* newptr) {
#if CMK_TLS_SWITCHING64
  asm volatile ("movq %0, %%fs:0x0\n\t"
                :
                : "r"(newptr));
#elif CMK_TLS_SWITCHING32
  asm volatile ("movl %0, %%gs:0x0\n\t"
                :
                : "r"(newptr));
#else
  fprintf(stderr, "TLS globals are not supported.");
  abort();
#endif
}

#endif
