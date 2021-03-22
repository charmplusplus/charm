
#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif
#ifndef __USE_GNU
# define __USE_GNU
#endif

#include "ampi_funcptr_loader.h"
#undef atexit

#include <string>
#include <atomic>
#include <vector>
#include <map>
#include <functional>

#if CMK_HAS_DL_ITERATE_PHDR
#include <link.h>

#if CMK_HAS_ELF_H
#include <elf.h>
#endif

#if CMK_HAS_MMAP
#include <sys/mman.h>
#endif

#ifndef PT_GNU_PROPERTY
#define PT_GNU_PROPERTY 0x6474e553
#endif
#endif

#define PIEGLOBALS_DEBUG 0

static std::atomic<size_t> rank_count{};

struct itemstruct
{
  size_t offset;
  size_t size;
  int flags;
};

struct objectstruct
{
  std::string name{};
  char * source = nullptr;
  size_t size = 0;
  size_t align = 0;
  std::vector<itemstruct> items{};
};

struct pieglobalsstruct
{
  int nodeleaderrank;
  SharedObject myexe;

  size_t numobjects = 0;
  std::vector<objectstruct> objects{};

  std::vector<std::tuple<ptrdiff_t, uintptr_t, size_t>> segment_allocation_offsets{};

  CmiIsomallocContext heap_context;
  std::vector<std::tuple<uintptr_t, size_t, size_t>> global_constructor_heap{};

  ampi_mainstruct mainstruct;
  ptrdiff_t main_c_diff, main_f_diff;

  void dealloc()
  {
    global_constructor_heap.clear();
  }
};

struct local_heap_entry
{
  void * addr;
  size_t size;
};

static int print(struct dl_phdr_info * info, size_t size, void * data)
{
  auto & numobjects = *(size_t *)data;
  if (numobjects > 0)
  {
    --numobjects;
    return 0;
  }

#if PIEGLOBALS_DEBUG >= 1
  CmiPrintf("pieglobals> Object - \"%s\" - %d segments\n", info->dlpi_name, info->dlpi_phnum);
#endif

  for (int j = 0; j < info->dlpi_phnum; j++)
  {
    const auto & phdr = info->dlpi_phdr[j];
    const int p_type = phdr.p_type;

    if (p_type == PT_TLS)
      CmiError("AMPI> Warning: Use of thread_local detected in pieglobals binary!\n");

#if PIEGLOBALS_DEBUG >= 1
    const char * type;
    switch (p_type)
    {
      case PT_LOAD: type = "PT_LOAD"; break;
      case PT_DYNAMIC: type = "PT_DYNAMIC"; break;
      case PT_INTERP: type = "PT_INTERP"; break;
      case PT_NOTE: type = "PT_NOTE"; break;
      case PT_SHLIB: type = "PT_SHLIB"; break;
      case PT_PHDR: type = "PT_PHDR"; break;
      case PT_TLS: type = "PT_TLS"; break;
      case PT_GNU_EH_FRAME: type = "PT_GNU_EH_FRAME"; break;
      case PT_GNU_STACK: type = "PT_GNU_STACK"; break;
      case PT_GNU_RELRO: type = "PT_GNU_RELRO"; break;
      case PT_GNU_PROPERTY: type = "PT_GNU_PROPERTY"; break;
      default: type = nullptr; break;
    }

    const uintptr_t start = (uintptr_t)info->dlpi_addr + phdr.p_vaddr;
    CmiPrintf("    %2d: [0x%012" PRIxPTR ", 0x%012" PRIxPTR "); memsz: 0x%07lx; align: 0x%04lx; flags: %c%c%c + 0x%x; ", j,
           start,
           start + phdr.p_memsz,
           phdr.p_memsz,
           phdr.p_align,
           (phdr.p_flags & PF_R) ? 'R' : '-',
           (phdr.p_flags & PF_W) ? 'W' : '-',
           (phdr.p_flags & PF_X) ? 'X' : '-',
           phdr.p_flags & ~(PF_R|PF_W|PF_X));
    if (type != nullptr)
      CmiPrintf("%s\n", type);
    else
      CmiPrintf("[other (0x%x)]\n", p_type);
#endif
  }

  return 0;
}

static int probe(struct dl_phdr_info * info, size_t size, void * data)
{
  auto & pieglobalsdata = *(pieglobalsstruct *)data;
  size_t & numobjects = pieglobalsdata.numobjects;
  if (numobjects > 0)
  {
    --numobjects;
    return 0;
  }

  pieglobalsdata.objects.emplace_back();
  objectstruct & obj = pieglobalsdata.objects.back();
  obj.name = info->dlpi_name;

  {
    void * objstart = (void *)(intptr_t)-1, * objend = nullptr;

    for (int j = 0; j < info->dlpi_phnum; j++)
    {
      const auto & phdr = info->dlpi_phdr[j];
      if (phdr.p_type != PT_LOAD)
        continue;

      void * start = (void *)(info->dlpi_addr + phdr.p_vaddr);
      void * end = (char *)start + phdr.p_memsz;
      if (objstart > start)
        objstart = start;
      if (objend < end)
        objend = end;
    }

    obj.source = (char *)objstart;
    obj.size = (char *)objend - (char *)objstart;
  }

  for (int j = 0; j < info->dlpi_phnum; j++)
  {
    const auto & phdr = info->dlpi_phdr[j];
    if (phdr.p_type != PT_LOAD)
      continue;

    obj.items.emplace_back();
    itemstruct & item = obj.items.back();

    auto start = (char *)(info->dlpi_addr + phdr.p_vaddr);
    if (start == obj.source)
      obj.align = phdr.p_align;

    item.offset = (char *)start - obj.source;
    item.size = phdr.p_memsz;
    item.flags = phdr.p_flags;
  }

  return 0;
}

static int count(struct dl_phdr_info * info, size_t size, void * data)
{
  auto & numobjects = *(size_t *)data;
  ++numobjects;
  return 0;
}

static size_t numobjects;
static pieglobalsstruct pieglobalsdata{};

static void pieglobalscleanup()
{
  if (pieglobalsdata.heap_context.opaque == nullptr)
    return;

  const CthThread th = TCharm::get()->getThread();

  CthInterceptionsDeactivatePush(th);

  CmiMemoryIsomallocContextActivate(pieglobalsdata.heap_context);
  dlclose(pieglobalsdata.myexe);

  CthInterceptionsDeactivatePop(th);

  CmiIsomallocContextDelete(pieglobalsdata.heap_context);

  pieglobalsdata.heap_context.opaque = nullptr;
}

static void pieglobalscleanupatexit()
{
  if (pieglobalsdata.heap_context.opaque == nullptr)
    return;

  CmiMemoryIsomallocContextActivate(pieglobalsdata.heap_context);
  dlclose(pieglobalsdata.myexe);
  CmiMemoryIsomallocContextActivate(CmiIsomallocContext{});

  CmiIsomallocContextDelete(pieglobalsdata.heap_context);

  pieglobalsdata.heap_context.opaque = nullptr;
}

void AMPI_Node_Setup(int numranks)
{
  if (CmiMyNode() == 0 && !quietModeRequested)
    CmiPrintf("AMPI> Using pieglobals privatization method.\n");

#if PIEGLOBALS_DEBUG >= 1
  if (!quietModeRequested)
    CmiPrintf("pieglobals> [%d][%d] AMPI_Node_Setup(%d)\n",
              CmiMyNode(), CmiMyPe(), numranks);
#endif

  // set up fake isomalloc contexts using n+1th slot
  constexpr int subdivisions = 2;
  int finalslotstart = numranks * subdivisions;
  int numsubdividedslots = (numranks+1) * subdivisions;
  CmiIsomallocContext heap_context = CmiIsomallocContextCreate(finalslotstart, numsubdividedslots);

  if (heap_context.opaque == nullptr)
    CkAbort("AMPI> pieglobals requires Isomalloc!");


  atexit(pieglobalscleanupatexit);

  AMPI_FuncPtr_Transport funcptrs{};
  AMPI_FuncPtr_Pack(&funcptrs);


  dl_iterate_phdr(count, &numobjects);

  static const char exe_suffix[] = STRINGIFY(CMK_POST_EXE);
  static const char suffix[] = STRINGIFY(CMK_USER_SUFFIX) "." STRINGIFY(CMK_SHARED_SUF);
  static constexpr size_t exe_suffix_len = sizeof(exe_suffix)-1;

  std::string binary_path{ampi_binary_path};
  if (exe_suffix_len > 0)
  {
    size_t pos = binary_path.length() - exe_suffix_len;
    if (!binary_path.compare(pos, exe_suffix_len, exe_suffix))
      binary_path.resize(pos);
  }
  binary_path += suffix;





  CmiIsomallocContextEnableRandomAccess(heap_context);
  CmiIsomallocContextEnableRecording(heap_context, 1);
  CmiMemoryIsomallocContextActivate(heap_context);

  SharedObject myexe = dlopen(binary_path.c_str(), RTLD_NOW|RTLD_LOCAL|RTLD_DEEPBIND);

  if (myexe == nullptr)
  {
    CkError("dlopen error: %s\n", dlerror());
    CkAbort("Could not open pieglobals user program!");
  }

  auto unpack = AMPI_FuncPtr_Unpack_Locate(myexe);
  if (unpack != nullptr)
    unpack(&funcptrs);

  pieglobalsdata.mainstruct = AMPI_Main_Get(myexe);

  CmiMemoryIsomallocContextActivate(CmiIsomallocContext{});
  CmiIsomallocContextEnableRecording(heap_context, 0);

  if (pieglobalsdata.mainstruct.c == nullptr && pieglobalsdata.mainstruct.f == nullptr)
    CkAbort("Could not find any AMPI entry points!");

  CmiIsomallocGetRecordedHeap(heap_context, pieglobalsdata.global_constructor_heap);
  pieglobalsdata.heap_context = heap_context;
  pieglobalsdata.myexe = myexe;


  if (CmiMyNode() == 0 && !quietModeRequested)
  {
    size_t mynumobjects = numobjects;
    dl_iterate_phdr(print, &mynumobjects);
  }

  pieglobalsdata.numobjects = numobjects;
  dl_iterate_phdr(probe, &pieglobalsdata);
  CmiAssert(!pieglobalsdata.objects.empty());

  atexit(ampiMarkAtexit);
}


using heap_map_t = std::map<uintptr_t, local_heap_entry, std::greater<uintptr_t>>;

static void replace_in_range(
  size_t myrank,
  void * range_start, void * range_end,
  const heap_map_t & my_heap_map,
  char * dstheapstart, CmiIsomallocRegion srcheap,
  const std::vector<char *> & segment_allocations,
  const char * from)
{
  for (void ** scan = (void **)range_start, ** end = (void **)range_end; scan < end; ++scan)
  {
    void * data = *scan;

    for (const auto & obj2 : pieglobalsdata.objects)
    {
      if (obj2.source <= data && data <= obj2.source + obj2.size)
      {
        char * allocation2 = segment_allocations[&obj2 - pieglobalsdata.objects.data()];
        void * out = allocation2 + ((char *)data - obj2.source);
#if PIEGLOBALS_DEBUG >= 2
        if (!quietModeRequested)
          CmiPrintf("pieglobals> [%d][%d][%zu] "
                    "0x%012" PRIxPTR ": found 0x%012" PRIxPTR ", writing 0x%012" PRIxPTR
                    " (in %s, to %s)\n",
                    CmiMyNode(), CmiMyPe(), myrank,
                    (uintptr_t)scan, (uintptr_t)data, (uintptr_t)out,
                    from, obj2.name.c_str());
#endif
        *scan = out;
      }
    }

#if 0
    // fast O(n) method, needs more work elsewhere to ensure deterministic heap reproduction
    if (srcheap.start <= data && data <= srcheap.end)
    {
      void * out = dstheapstart + ((char *)data - (char *)srcheap.start);
#if PIEGLOBALS_DEBUG >= 2
      if (!quietModeRequested)
        CmiPrintf("pieglobals> [%d][%d][%zu] "
                  "0x%012" PRIxPTR ": found 0x%012" PRIxPTR ", writing 0x%012" PRIxPTR
                  " (in %s, to heap)\n",
                  CmiMyNode(), CmiMyPe(), myrank,
                  (uintptr_t)scan, (uintptr_t)data, (uintptr_t)out,
                  from);
#endif
      *scan = out;
    }
#else
    // O(log n) method
    auto iter = my_heap_map.lower_bound((uintptr_t)data);
    if (iter != my_heap_map.end() && /* (char *)iter->first <= data && */ data <= (char *)iter->first + iter->second.size)
    {
      void * out = (char *)iter->second.addr + ((char *)data - (char *)iter->first);
#if PIEGLOBALS_DEBUG >= 2
      if (!quietModeRequested)
        CmiPrintf("pieglobals> [%d][%d][%zu] "
                  "0x%012" PRIxPTR ": found 0x%012" PRIxPTR ", writing 0x%012" PRIxPTR
                  " (in %s, to heap)\n",
                  CmiMyNode(), CmiMyPe(), myrank,
                  (uintptr_t)scan, (uintptr_t)data, (uintptr_t)out,
                  from);
#endif
      *scan = out;
    }
#endif
  }
}

static constexpr ptrdiff_t null_funcptr_diff = -1;

template <typename T>
static void calc_funcptr_diff(ptrdiff_t & diff, T funcptr, const void * objsource, const char * objend, const void * emptyheapstart, const char * allocation)
{
  auto ptr = (const char *)funcptr;
  if (ptr == nullptr)
  {
    diff = null_funcptr_diff;
    return;
  }

  if (ptr < objsource || objend <= ptr)
    return;

  diff = ptr - (const char *)objsource + (allocation - (const char *)emptyheapstart);
}

void AMPI_Rank_Setup(int myrank, int numranks, CmiIsomallocContext ctx)
{
  const size_t localrank = rank_count++;

#if PIEGLOBALS_DEBUG >= 1
  if (!quietModeRequested)
    CmiPrintf("pieglobals> [%d][%d][%zu] AMPI_Rank_Setup(%d, %d, %p)\n",
              CmiMyNode(), CmiMyPe(), localrank, myrank, numranks, ctx.opaque);
#endif

  if (ctx.opaque == nullptr)
    CkAbort("pieglobals requires Isomalloc!");

  if (localrank == 0)
    pieglobalsdata.nodeleaderrank = myrank;

  std::vector<char *> segment_allocations;
  for (const auto & obj : pieglobalsdata.objects)
  {
#if PIEGLOBALS_DEBUG >= 1
    if (!quietModeRequested)
      CmiPrintf("pieglobals> [%d][%d][%d] Processing %s\n",
                CmiMyNode(), CmiMyPe(), myrank, obj.name.c_str());
#endif

    auto allocation = (char *)CmiIsomallocContextPermanentAllocAlign(ctx, obj.align, obj.size);
    segment_allocations.emplace_back(allocation);

    for (const auto & item : obj.items)
    {
      char * addr = allocation + item.offset;
      char * src = obj.source + item.offset;
      size_t len = item.size;

#if PIEGLOBALS_DEBUG >= 1
      if (!quietModeRequested)
        CmiPrintf("pieglobals> [%d][%d][%d] [0x%012" PRIxPTR ", 0x%012" PRIxPTR ") --> [0x%012" PRIxPTR ", 0x%012" PRIxPTR ")\n",
                  CmiMyNode(), CmiMyPe(), myrank, (uintptr_t)src, (uintptr_t)src + len, (uintptr_t)addr, (uintptr_t)addr + len);
#endif

      memcpy(addr, src, len);
    }
  }

  const CmiIsomallocRegion heapBefore = CmiIsomallocContextGetUsedExtent(ctx);
  const auto heapstart = (char *)heapBefore.start;
  const auto mallocstart = (char *)heapBefore.end;

  if (localrank == 0)
  {
    for (const auto & obj : pieglobalsdata.objects)
    {
      char * allocation = segment_allocations[&obj - pieglobalsdata.objects.data()];

      pieglobalsdata.segment_allocation_offsets.emplace_back(allocation - heapstart, (uintptr_t)obj.source, obj.size);

      for (const auto & item : obj.items)
      {
        char * addr = allocation + item.offset;
        char * src = obj.source + item.offset;
        size_t len = item.size;

        // calculate the offset of the function pointer into the isomalloc heap
        // will be isomorphic for all ranks
        // avoids needing to store per-rank segment information between AMPI_Rank_Setup and main
        char * end = src + len;
        calc_funcptr_diff(pieglobalsdata.main_c_diff, pieglobalsdata.mainstruct.c, src, end, heapstart, allocation);
        calc_funcptr_diff(pieglobalsdata.main_f_diff, pieglobalsdata.mainstruct.f, src, end, heapstart, allocation);
      }
    }
  }


  CmiIsomallocContextEnableRandomAccess(ctx);

  heap_map_t my_heap_map;
  for (const auto & entry : pieglobalsdata.global_constructor_heap)
  {
    const auto src = (void *)std::get<0>(entry);
    const size_t size = std::get<1>(entry);
    const size_t align = std::get<2>(entry);

    void * dst = align == 0
                 ? CmiIsomallocContextMalloc(ctx, size)
                 : CmiIsomallocContextMallocAlign(ctx, align, size);
    CmiEnforce(dst != nullptr);

    memcpy(dst, src, size);

    my_heap_map.emplace((uintptr_t)src, local_heap_entry{dst, size});
  }

  // apply fixups here

  const CmiIsomallocRegion srcheap = CmiIsomallocContextGetUsedExtent(pieglobalsdata.heap_context);

  for (const auto & obj : pieglobalsdata.objects)
  {
    char * allocation = segment_allocations[&obj - pieglobalsdata.objects.data()];

    for (const auto & item : obj.items)
    {
      char * addr = allocation + item.offset;
      void * src = obj.source + item.offset;
      size_t len = item.size;

      // scan the globals for pointers to code, globals, or the global constructor heap
      if (!(item.flags & PF_X))
      {
        replace_in_range(myrank, addr, addr + len, my_heap_map, mallocstart, srcheap, segment_allocations, obj.name.c_str());
      }
    }
  }

  // replace_in_range(myrank, heapBefore.end, heap.end, my_heap_map, mallocstart, srcheap, segment_allocations, "FULL HEAP - BAD");

#if 1
  // scan the global constructor heap
  for (const auto & entry : my_heap_map)
  {
    const local_heap_entry & dst = entry.second;
    replace_in_range(myrank, dst.addr, (char *)dst.addr + dst.size, my_heap_map, mallocstart, srcheap, segment_allocations, "heap");
  }
#endif

  for (const auto & obj : pieglobalsdata.objects)
  {
    char * allocation = segment_allocations[&obj - pieglobalsdata.objects.data()];
    for (const auto & item : obj.items)
    {
      char * addr = allocation + item.offset;
      size_t len = item.size;

      int prot = 0;
      if (item.flags & PF_R)
        prot |= PROT_READ;
      if (item.flags & PF_W)
        prot |= PROT_WRITE;
      if (item.flags & PF_X)
        prot |= PROT_EXEC;

      CmiIsomallocContextProtect(ctx, addr, CMIALIGN(len, obj.align), prot);
    }
  }
}

// separate function so that setting a breakpoint is straightforward
static inline int ampi_pieglobals(int argc, char ** argv)
{
  const CthThread th = TCharm::get()->getThread();
  auto ctx = CmiIsomallocGetThreadContext(th);
  const CmiIsomallocRegion heap = CmiIsomallocContextGetUsedExtent(ctx);
  const auto heapstart = (char *)heap.start;

  if (pieglobalsdata.main_c_diff != null_funcptr_diff)
  {
    auto newptr = (ampi_maintype)(heapstart + pieglobalsdata.main_c_diff);
    return newptr(argc, argv);
  }
  else if (pieglobalsdata.main_f_diff != null_funcptr_diff)
  {
    auto newptr = (ampi_fmaintype)(heapstart + pieglobalsdata.main_f_diff);
    newptr();
    return 0;
  }

  return 1;
}

int main(int argc, char ** argv)
{
  const int myrank = TCHARM_Element();

  TCHARM_Barrier();

  if (myrank == pieglobalsdata.nodeleaderrank)
  {
    pieglobalsdata.dealloc();

#if 0
    // clean up system-provided segments and global constructor heap BEFORE running
    // benefit: saves memory
    // benefit: replaces silent lapses in privatization with segfaults
    // benefit: avoids crashes during global destructors
    // detriment: no longer have original code in memory to cross-reference in debugger
    // showstopper: causes crashes inside libdl during program execution
    pieglobalscleanup();
#endif
  }

  TCHARM_Barrier();

  return ampi_pieglobals(argc, argv);
}

// for debugging purposes
// GDB: `call pieglobalsfind($rip)` or `call pieglobalsfind((void *)0x...)`
void * pieglobalsfind(void * ptr)
{
  const CthThread th = TCharm::get()->getThread();
  auto ctx = CmiIsomallocGetThreadContext(th);
  const CmiIsomallocRegion heap = CmiIsomallocContextGetUsedExtent(ctx);

  if (ptr < heap.start || heap.end <= ptr)
  {
    fprintf(stderr, "pointer not in range of rank's data\n");
    return nullptr;
  }

  ptrdiff_t offset = (char *)ptr - (char *)heap.start;

  for (const auto & seg : pieglobalsdata.segment_allocation_offsets)
  {
    ptrdiff_t allocation_offset = std::get<0>(seg);
    uintptr_t src = std::get<1>(seg);
    size_t size = std::get<2>(seg);

    if (allocation_offset <= offset && offset < allocation_offset + size)
    {
      uintptr_t out = src + (offset - allocation_offset);
      return (void *)out;
    }
  }

  fprintf(stderr, "other rank data (call stack or heap)\n");
  return nullptr;
}
