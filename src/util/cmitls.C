
#include <stdio.h>

#include "converse.h"
#include "cmitls.h"

#include <string.h>
#include <stdlib.h>
#if CMK_HAS_MALLOC_H
#include <malloc.h>
#endif

#include "memory-isomalloc.h"

#if CMK_HAS_TLS_VARIABLES

extern int quietModeRequested;

/* These macros are needed for:
 * dlfcn.h: RTLD_DEFAULT
 * link.h: dl_iterate_phdr
 */
#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif
#ifndef __USE_GNU
# define __USE_GNU
#endif


// ----- TLS segment pointer access -----

/*
 * For a description of the system TLS implementations this file works with, see:
 * "ELF Handling For Thread-Local Storage"
 * https://www.akkadia.org/drepper/tls.pdf
 * Of note are sections 3.4.2 (IA-32, a.k.a. x86) and 3.4.6 (x86-64).
 */

extern "C" {
void* getTLS();
void setTLS(void*);
void* swapTLS(void*);
}

#if CMK_TLS_SWITCHING_X86_64
# define CMK_TLS_X86_MOV "movq"
# ifdef __APPLE__
#  define CMK_TLS_X86_REG "gs"
# else
#  define CMK_TLS_X86_REG "fs"
# endif
#elif CMK_TLS_SWITCHING_X86
# define CMK_TLS_X86_MOV "movl"
# define CMK_TLS_X86_REG "gs"
#else
# define CMK_TLS_SWITCHING_UNAVAILABLE
#endif

void* getTLS()
{
#ifdef CMK_TLS_X86_MOV
  void* ptr;
  asm volatile (CMK_TLS_X86_MOV " %%" CMK_TLS_X86_REG ":0x0, %0\n"
                : "=&r"(ptr));
  return ptr;
#else
  return nullptr;
#endif
}

void setTLS(void* newptr)
{
#ifdef CMK_TLS_X86_MOV
  asm volatile (CMK_TLS_X86_MOV " %0, %%" CMK_TLS_X86_REG ":0x0\n\t"
                :
                : "r"(newptr));
#endif
}

void* swapTLS(void* newptr)
{
  void* oldptr = getTLS();
  setTLS(newptr);
  return oldptr;
}


// ----- TLS segment size determination -----

#if CMK_HAS_DL_ITERATE_PHDR

# include <link.h>

static inline void CmiTLSStatsInit(void)
{
}

static int count_tls_sizes(struct dl_phdr_info* info, size_t size, void* data)
{
  size_t i;
  tlsseg_t* t = (tlsseg_t*)data;

  for (i = 0; i < info->dlpi_phnum; i++)
  {
    const ElfW(Phdr) * hdr = &info->dlpi_phdr[i];
    if (hdr->p_type == PT_TLS)
    {
      t->size += hdr->p_memsz;
      if (t->align < hdr->p_align)
        t->align = hdr->p_align;
    }
  }

  return 0;
}

static void populateTLSSegStats(tlsseg_t * t)
{
  t->size = 0;
  t->align = 0;
  dl_iterate_phdr(count_tls_sizes, t); /* count all PT_TLS sections */
}

#elif defined __APPLE__

// parts adapted from threadLocalVariables.c in dyld

# include <mach-o/dyld.h>
# include <mach-o/nlist.h>

#if __LP64__
  typedef struct mach_header_64 macho_header;
# define MACHO_HEADER_MAGIC MH_MAGIC_64
# define LC_SEGMENT_COMMAND LC_SEGMENT_64
  typedef struct segment_command_64 macho_segment_command;
  typedef struct section_64 macho_section;
  typedef struct nlist_64 macho_nlist;
#else
  typedef struct mach_header macho_header;
# define MACHO_HEADER_MAGIC MH_MAGIC
# define LC_SEGMENT_COMMAND LC_SEGMENT
  typedef struct segment_command macho_segment_command;
  typedef struct section macho_section;
  typedef struct nlist macho_nlist;
#endif

static size_t GetTLVSizeFromMachOHeader()
{
  size_t totalsize = 0;

  for (uint32_t c = 0; c < _dyld_image_count(); ++c)
  {
    const struct mach_header * const mh_orig = _dyld_get_image_header(c);
    if (mh_orig == nullptr)
      continue;

    CmiEnforce(mh_orig->magic == MACHO_HEADER_MAGIC);
    const auto mh = (const macho_header *)mh_orig;
    const auto mh_addr = (const char *)mh;

    const uint8_t * start = nullptr;
    unsigned long size = 0;
    intptr_t slide = 0;
    bool slideComputed = false;

    uint64_t text_vmaddr = 0, linkedit_vmaddr = 0, linkedit_fileoff = 0;

    const uint32_t cmd_count = mh->ncmds;
    const auto cmds = (const struct load_command *)(mh_addr + sizeof(macho_header));
    const struct load_command * cmd = cmds;
    for (uint32_t i = 0; i < cmd_count; ++i)
    {
      const auto lc_type = cmd->cmd & ~LC_REQ_DYLD;
      if (lc_type == LC_SEGMENT_COMMAND)
      {
        const auto seg = (const macho_segment_command *)cmd;

        if (!slideComputed && seg->filesize != 0)
        {
          slide = (uintptr_t)mh - seg->vmaddr;
          slideComputed = true;
        }

        if (strcmp(seg->segname, SEG_TEXT) == 0)
          text_vmaddr = seg->vmaddr;
        else if (strcmp(seg->segname, SEG_LINKEDIT) == 0)
        {
          linkedit_vmaddr = seg->vmaddr;
          linkedit_fileoff = seg->fileoff;
        }

        // look for TLV sections, used by Apple's clang

        const auto sectionsStart = (const macho_section *)((const char *)seg + sizeof(macho_segment_command));
        const auto sectionsEnd = sectionsStart + seg->nsects;
        for (auto sect = sectionsStart; sect < sectionsEnd; ++sect)
        {
          const auto section_type = sect->flags & SECTION_TYPE;
          if (section_type == S_THREAD_LOCAL_ZEROFILL || section_type == S_THREAD_LOCAL_REGULAR)
          {
            if (start == nullptr)
            {
              // first of N contiguous TLV template sections: record as if this were the only section
              start = (const uint8_t *)(sect->addr + slide);
              size = sect->size;
            }
            else
            {
              // non-first of N contiguous TLV template sections: accumulate values
              const auto newEnd = (const uint8_t *)(sect->addr + slide + sect->size);
              size = newEnd - start;
            }
          }
        }
      }
      else if (lc_type == LC_SYMTAB && text_vmaddr)
      {
        // look through all symbols for GNU emutls, used by GCC

        const auto symcmd = (const struct symtab_command *)cmd;
        auto strtab = (const char *)(linkedit_vmaddr + (symcmd->stroff - linkedit_fileoff) + slide);
        auto symtab = (const macho_nlist *)(linkedit_vmaddr + (symcmd->symoff - linkedit_fileoff) + slide);

        for (const macho_nlist * nl = symtab, * const nl_end = symtab + symcmd->nsyms; nl < nl_end; ++nl)
        {
          if ((nl->n_type & (N_TYPE & N_SECT)) != N_SECT)
            continue;

          static const char emutls_prefix[] = "___emutls_v.";
          const char * const symname = strtab + nl->n_un.n_strx;
          if (strncmp(symname, emutls_prefix, sizeof(emutls_prefix)-1) == 0)
          {
            const auto addr = (const size_t *)(nl->n_value + slide);
            const size_t symsize = *addr;
            totalsize += symsize;
          }
        }
      }

      cmd = (const struct load_command *)(((const char *)cmd) + cmd->cmdsize);
    }

    totalsize += size;
  }

  return totalsize;
}

static size_t CmiTLSSize;

static inline void CmiTLSStatsInit(void)
{
  // calculate the TLS size once at startup
  CmiTLSSize = GetTLVSizeFromMachOHeader();
}

static void populateTLSSegStats(tlsseg_t * t)
{
  // fill the struct with the cached size
  t->size = CmiTLSSize;
  // Apple uses alignment by 16
  t->align = 16;
}

#elif CMK_HAS_ELF_H && CMK_DLL_USE_DLOPEN && CMK_HAS_RTLD_DEFAULT

# include <dlfcn.h>
# define CMK_TLS_NO_SHARED

static void* CmiTLSExecutableStart;

static inline Addr getCodeSegAddr()
{
  return (Addr) CmiTLSExecutableStart;
}

static inline Ehdr* getELFHeader()
{
  return (Ehdr*) getCodeSegAddr();
}

static inline Phdr* getProgramHeader(Ehdr* ehdr)
{
  return (Phdr*)((char *)ehdr + ehdr->e_phoff);
}

Phdr* getTLSPhdrEntry()
{
  int phnum, i;
  Ehdr* elfHeader;
  Phdr* progHeader;

  elfHeader = getELFHeader();
  if (elfHeader == NULL)
    return NULL;

  phnum = elfHeader->e_phnum;
  progHeader = getProgramHeader(elfHeader);
  for (i = 0; i < phnum; i++)
  {
    if (progHeader[i].p_type == PT_TLS)
    {
#if CMK_ERROR_CHECKING
      /* sanity check */
      /* align is power of 2 */
      int align = progHeader[i].p_align;
      CmiAssert(align > 0 && (align & (align-1)) == 0);
      /* total size is not less than the size of .tdata (initializer data) */
      CmiAssert(progHeader[i].p_memsz >= progHeader[i].p_filesz);
#endif
      return &progHeader[i];
    }
  }
  return NULL;
}

static void CmiTLSStatsInit()
{
  /* Use dynamic linking in case Charm++ shared objects are used by a binary lacking
   * conv-static.o, such as in the case of Charm4py. */
  void** pCmiExecutableStart = (void**)dlsym(RTLD_DEFAULT, "CmiExecutableStart");
  if (pCmiExecutableStart != NULL)
    CmiTLSExecutableStart = *pCmiExecutableStart;
  else
    CmiPrintf("Charm++> Error: \"CmiExecutableStart\" symbol not found. -tlsglobals disabled.\n");
}

static void populateTLSSegStats(tlsseg_t * t)
{
  Phdr* phdr = getTLSPhdrEntry();
  if (phdr != NULL)
  {
    t->align = phdr->p_align;
    t->size = phdr->p_memsz;
  }
  else
  {
    t->size = 0;
    t->align = 0;
  }
}

#else

static inline void CmiTLSStatsInit()
{
}

static void populateTLSSegStats(tlsseg_t * t)
{
  t->size = 0;
}

#endif


// ----- CmiTLS implementation -----

void CmiTLSInit()
{
#ifdef CMK_TLS_SWITCHING_UNAVAILABLE
  CmiAbort("TLS globals are not supported.");
#else
  if (CmiMyRank() == 0)
  {
    if (!quietModeRequested && CmiMyPe() == 0)
    {
      CmiPrintf("Charm++> -tlsglobals enabled for privatization of thread-local variables.\n");
#ifdef CMK_TLS_NO_SHARED
      CmiPrintf("Charm++> Warning: Unable to examine TLS segments of shared objects.\n");
#endif
    }

    CmiTLSStatsInit();
  }
#endif
}

void allocNewTLSSeg(tlsseg_t* t, CthThread th)
{
  populateTLSSegStats(t);

  if (t->size > 0)
  {
    t->size = CMIALIGN(t->size, t->align);
    t->memseg = (Addr)CmiIsomallocMallocAlignForThread(th, t->align, t->size);
    memcpy((void*)t->memseg, (char *)getTLS() - t->size, t->size);
    t->memseg = (Addr)( ((char *)(t->memseg)) + t->size );
    /* printf("[%d] 2 ALIGN %d MEM %p SIZE %d\n", CmiMyPe(), t->align, t->memseg, t->size); */
  }
  else
  {
    /* since we don't have a PT_TLS section to copy, keep whatever the system gave us */
    t->memseg = (Addr)getTLS();
  }
}

void switchTLS(tlsseg_t* cur, tlsseg_t* next)
{
  cur->memseg = (Addr)swapTLS((void*)next->memseg);
}

void currentTLS(tlsseg_t* cur)
{
  cur->memseg = (Addr)getTLS();
}

#endif
