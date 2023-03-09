/*
 * Heap Record-Replay
 * By Evan Ramos
 *
 * Generates an MPI program that repeats the exact sequence of heap operations observed
 * on each PE. Can be used for debugging and performance analysis of memory allocators
 * as well as heap behavior of user codes.
 */

#include <cstdint>
#include <cstdio>
#include <unordered_map>

// raw string literals containing the output program

static const char prologue[] =
R"code(
#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"

int main(int argc, char ** argv)
{
  MPI_Init(&argc, &argv);

  int p;
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  if (%d != p)
    puts("This recording was intended for %d ranks.");

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  switch (rank)
  {
)code";

static const char body[] =
R"code(
    case %d:
    {
#include "heap-replay-%d.cpp"
      break;
    }
)code";

static const char epilogue[] =
R"code(
    default:
      break;
  }

  MPI_Finalize();
  return 0;
}
)code";

// end raw string literals

static char initialized;
struct MemoryRecord {
  std::unordered_map<uintptr_t, unsigned long long> ptrs;
  FILE * output;
  unsigned long long count;
  char guard;
};
CpvStaticDeclare(MemoryRecord, state);

static void meta_init(char **argv)
{
  CpvInitialize(MemoryRecord, state);

  const int mype = CmiMyPe();

  char outputname[64];
  snprintf(outputname, sizeof(outputname), "heap-replay-%d.cpp", mype);
  FILE * output = fopen(outputname, "w");
  if (output == nullptr)
    CmiAbort("Could not open heap-replay-%d.cpp!", mype);
  CpvAccess(state).output = output;

  if (CmiMyRank() == 0)
    initialized = 1;

  CmiNodeAllBarrier();

  if (mype == 0)
  {
    const int numpes = CmiNumPes();

    FILE * mainsrc = fopen("heap-replay.cpp", "w");
    if (mainsrc == nullptr)
      CmiAbort("Could not open heap-replay.cpp!");
    fprintf(mainsrc, prologue, numpes, numpes);
    for (int pe = 0; pe < numpes; ++pe)
      fprintf(mainsrc, body, pe, pe);
    fprintf(mainsrc, epilogue);
    fclose(mainsrc);
  }
}

static void *meta_malloc(size_t size)
{
  void *ret = mm_malloc(size);

  if (initialized && !CpvAccess(state).guard)
  {
    MemoryRecord & data = CpvAccess(state);
    data.guard = 1;

    fprintf(data.output, "void * ptr_%llu = malloc(%zu);\n", data.count, size);
    data.ptrs.emplace((uintptr_t)ret, data.count);
    ++data.count;

    data.guard = 0;
  }

  return ret;
}

static void meta_free(void *mem)
{
  if (initialized && !CpvAccess(state).guard)
  {
    MemoryRecord & data = CpvAccess(state);
    data.guard = 1;

    if (mem == nullptr)
      fprintf(data.output, "free(nullptr);\n");
    else
    {
      auto iter = data.ptrs.find((uintptr_t)mem);
      if (iter != data.ptrs.end())
      {
        fprintf(data.output, "free(ptr_%llu);\n", iter->second);
        data.ptrs.erase(iter);
      }
#if 0
      else // causes an abort during global variable destructors
        CmiAbort("Got free(%p) without recording a matching allocation!", mem);
#endif
    }

    data.guard = 0;
  }

  mm_free(mem);
}

static void *meta_calloc(size_t nelem, size_t size)
{
  void *ret = mm_calloc(nelem, size);

  if (initialized && !CpvAccess(state).guard)
  {
    MemoryRecord & data = CpvAccess(state);
    data.guard = 1;

    fprintf(data.output, "void * ptr_%llu = calloc(%zu, %zu);\n", data.count, nelem, size);
    data.ptrs.emplace((uintptr_t)ret, data.count);
    ++data.count;

    data.guard = 0;
  }

  return ret;
}

static void meta_cfree(void *mem)
{
  if (initialized && !CpvAccess(state).guard)
  {
    MemoryRecord & data = CpvAccess(state);
    data.guard = 1;

    if (mem == nullptr)
      fprintf(data.output, "cfree(nullptr);\n");
    else
    {
      auto iter = data.ptrs.find((uintptr_t)mem);
      if (iter != data.ptrs.end())
      {
        fprintf(data.output, "cfree(ptr_%llu);\n", iter->second);
        data.ptrs.erase(iter);
      }
#if 0
      else
        CmiAbort("Got cfree(%p) without recording a matching allocation!", mem);
#endif
    }

    data.guard = 0;
  }
  mm_cfree(mem);
}

static void *meta_realloc(void *mem, size_t size)
{
  void *ret = mm_realloc(mem, size);

  if (initialized && !CpvAccess(state).guard)
  {
    MemoryRecord & data = CpvAccess(state);
    data.guard = 1;

    if (mem == nullptr)
      fprintf(data.output, "void * ptr_%llu = realloc(nullptr, %zu);\n", data.count, size);
    else
    {
      auto iter = data.ptrs.find((uintptr_t)mem);
      if (iter != data.ptrs.end())
      {
        fprintf(data.output, "void * ptr_%llu = realloc(ptr_%llu, %zu);\n", data.count, iter->second, size);
        data.ptrs.erase(iter);
      }
#if 0
      else
        CmiAbort("Got realloc(%p, %zu) without recording a matching allocation!", mem, size);
#endif
    }
    ++data.count;

    data.guard = 0;
  }

  return ret;
}

static void *meta_memalign(size_t align, size_t size)
{
  void *ret = mm_memalign(align, size);

  if (initialized && !CpvAccess(state).guard)
  {
    MemoryRecord & data = CpvAccess(state);
    data.guard = 1;

    fprintf(data.output, "void * ptr_%llu = memalign(%zu, %zu);\n", data.count, align, size);
    data.ptrs.emplace((uintptr_t)ret, data.count);
    ++data.count;

    data.guard = 0;
  }

  return ret;
}

static int meta_posix_memalign(void **outptr, size_t align, size_t size)
{
  int ret = mm_posix_memalign(outptr, align, size);

  if (initialized && !CpvAccess(state).guard)
  {
    MemoryRecord & data = CpvAccess(state);
    data.guard = 1;

    fprintf(data.output, "void * ptr_%llu;\nposix_memalign(&ptr_%llu, %zu, %zu);\n", data.count, data.count, align, size);
    data.ptrs.emplace((uintptr_t)*outptr, data.count);
    ++data.count;

    data.guard = 0;
  }

  return ret;
}

static void *meta_aligned_alloc(size_t align, size_t size)
{
  void *ret = mm_aligned_alloc(align, size);

  if (initialized && !CpvAccess(state).guard)
  {
    MemoryRecord & data = CpvAccess(state);
    data.guard = 1;

    fprintf(data.output, "void * ptr_%llu = aligned_alloc(%zu, %zu);\n", data.count, align, size);
    data.ptrs.emplace((uintptr_t)ret, data.count);
    ++data.count;

    data.guard = 0;
  }

  return ret;
}

static void *meta_valloc(size_t size)
{
  void *ret = mm_valloc(size);

  if (initialized && !CpvAccess(state).guard)
  {
    MemoryRecord & data = CpvAccess(state);
    data.guard = 1;

    fprintf(data.output, "void * ptr_%llu = valloc(%zu);\n", data.count, size);
    data.ptrs.emplace((uintptr_t)ret, data.count);
    ++data.count;

    data.guard = 0;
  }

  return ret;
}

static void *meta_pvalloc(size_t size)
{
  void *ret = mm_pvalloc(size);

  if (initialized && !CpvAccess(state).guard)
  {
    MemoryRecord & data = CpvAccess(state);
    data.guard = 1;

    fprintf(data.output, "void * ptr_%llu = pvalloc(%zu);\n", data.count, size);
    data.ptrs.emplace((uintptr_t)ret, data.count);
    ++data.count;

    data.guard = 0;
  }

  return ret;
}

// Filter out RTS allocations which can be freed from PEs other than their originator.
#define CMK_MEMORY_HAS_NOMIGRATE
void *malloc_nomigrate(size_t size)
{
  return mm_malloc(size);
}
void free_nomigrate(void *mem)
{
  mm_free(mem);
}
