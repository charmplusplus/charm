
#include <stdio.h>

#include "converse.h"
#include "cmitls.h"

#if CMK_HAS_ELF_H && CMK_HAS_TLS_VARIABLES

extern void* __executable_start;

static Addr getCodeSegAddr() {
  return (Addr) &__executable_start;
}

static Ehdr* getELFHeader() {
  return (Ehdr*) getCodeSegAddr();
}

static Phdr* getProgramHeader() {
  Ehdr* ehdr = getELFHeader();
  return (Phdr*)((char *)ehdr + ehdr->e_phoff);
}

Phdr* getTLSPhdrEntry() {
  int phnum, i;
  Ehdr* elfHeader;
  Phdr* progHeader;

  elfHeader = getELFHeader();
  phnum = elfHeader->e_phnum;
  progHeader = getProgramHeader();
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
    t->memseg = (Addr)( ((void*)(t->memseg)) + t->size );
    /* printf("[%d] 2 ALIGN %d MEM %p SIZE %d\n", CmiMyPe(), t->align, t->memseg, t->size); */
  } else {
    t->memseg = (Addr)NULL;
    t->size = 0;
    t->align = 0;
  }
}

void switchTLS(tlsseg_t* , tlsseg_t* ) __attribute__((optimize(0)));
void currentTLS(tlsseg_t*) __attribute__((optimize(0)));

/* void allocNewTLSSegEmpty(tlsseg_t* t) {
 *   void* aux = CmiIsomallocAlign(t->align, t->size);
 *     t->memseg = (Addr) (aux + t->size);
 *     } */

void currentTLS(tlsseg_t* cur) {
#if CMK_TLS_SWITCHING64
  asm volatile ("movq %%fs:0x0, %0\n\t"
                : "=r"(cur->memseg));
#elif CMK_TLS_SWITCHING32
  asm volatile ("movl %%gs:0x0, %0\n\t"
                : "=r"(cur->memseg));
#else
  fprintf(stderr, "TLS globals are not supported.");
  abort();
#endif
}

void switchTLS(tlsseg_t* cur, tlsseg_t* next) {
#if CMK_TLS_SWITCHING64
#if 0
  asm volatile ("movq %%fs:0x0, %0\n\t"
                : "=r"(cur->memseg));
  if (cur->memseg == next->memseg) return;      /* same */
  asm volatile ("movq %0, %%fs:0x0\n\t"
                :
                : "r"(next->memseg));
#else
  asm volatile ("movq %%fs:0x0, %0\n\t"
                "movq %1, %%fs:0x0\n\t"
                : "=r"(cur->memseg)
                : "r"(next->memseg));
#endif
#elif CMK_TLS_SWITCHING32
  asm volatile ("movl %%gs:0x0, %0\n\t"
                "movl %1, %%gs:0x0\n\t"
                : "=r"(cur->memseg)
                : "r"(next->memseg));
#else
  fprintf(stderr, "TLS globals are not supported.");
  abort();
#endif
}


#endif
