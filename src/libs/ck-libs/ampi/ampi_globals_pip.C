
#include "ampi_funcptr_loader.h"

#include <string>
#include <atomic>

static std::atomic<size_t> rank_count{};

int main(int argc, char ** argv)
{
  const size_t myrank = rank_count++;
  if (CmiMyNode() == 0 && myrank == 0 && !quietModeRequested)
    CmiPrintf("AMPI> Using pipglobals privatization method.\n");

  SharedObject myexe;

  // open the user binary for this rank in a unique namespace
  {
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

    const Lmid_t lmid = myrank == 0 ? LM_ID_BASE : LM_ID_NEWLM;
    myexe = dlmopen(lmid, binary_path.c_str(), RTLD_NOW|RTLD_LOCAL);
  }

  if (myexe == nullptr)
  {
    CkError("dlmopen error: %s\n", dlerror());
    CkAbort("Could not open pipglobals user program!");
  }

  return AMPI_FuncPtr_Loader(myexe, argc, argv);
}
