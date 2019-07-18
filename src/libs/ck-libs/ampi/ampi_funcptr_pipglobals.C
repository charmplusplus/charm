
#include "ampi_funcptr_loader.h"

#include <string>
#include <atomic>

static std::atomic<size_t> rank_count{};

int main(int argc, char ** argv)
{
  SharedObject myexe;

  // open the user binary for this rank in a unique namespace
  {
    static const char FUNCPTR_SHIM_SUFFIX[] = ".user";

    std::string binary_path{ampi_binary_path};
    binary_path += FUNCPTR_SHIM_SUFFIX;

    const Lmid_t lmid = rank_count++ == 0 ? LM_ID_BASE : LM_ID_NEWLM;
    myexe = dlmopen(lmid, binary_path.c_str(), RTLD_NOW|RTLD_LOCAL);
  }

  if (myexe == nullptr)
  {
    CkError("dlmopen error: %s\n", dlerror());
    CkAbort("Could not open pipglobals user program!");
  }

  return AMPI_FuncPtr_Loader(myexe, argc, argv);
}
