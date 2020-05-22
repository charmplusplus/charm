
#include "ampi_funcptr_loader.h"

#if defined _WIN32
# ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
# endif
# ifndef NOMINMAX
#  define NOMINMAX
# endif
# include <windows.h>
# include <io.h>
# define access _access
# ifndef R_OK
#  define R_OK 4
# endif
#elif defined __APPLE__
# include <unistd.h>
# include <copyfile.h>
# include <errno.h>
#else
# include <unistd.h>
# include <sys/types.h>
# include <sys/wait.h>
# include <errno.h>
#endif

#include <string>
#include <atomic>

static void fs_copy(const char * src, const char * dst)
{
  static const char abortmsg[] = "Could not copy fsglobals user program!";

#if defined _WIN32
  BOOL ret = CopyFile(src, dst, true);
  if (ret == 0)
  {
    CkError("ERROR> CopyFile(): %d\n", (int)GetLastError());
    CkAbort(abortmsg);
  }
#elif defined __APPLE__
  int ret = copyfile(src, dst, 0, COPYFILE_ALL);
  if (ret < 0)
  {
    CkError("ERROR> copyfile(): %d %s\n", ret, strerror(errno));
    CkAbort(abortmsg);
  }
#else
  pid_t pid = fork();
  if (pid == 0)
  {
    execl("/bin/cp", "/bin/cp", src, dst, NULL);
    CkError("ERROR> execl(): %s\n", strerror(errno));
    CkAbort(abortmsg);
  }
  else if (pid < 0)
  {
    CkError("ERROR> fork(): %s\n", strerror(errno));
    CkAbort(abortmsg);
  }
  else
  {
    int status;
    pid_t ws = waitpid(pid, &status, 0);
    if (ws == -1)
      CkError("ERROR> waitpid(): %s\n", strerror(errno));
  }
#endif
}

static std::atomic<size_t> rank_count{};

int main(int argc, char ** argv)
{
  SharedObject myexe;

  // copy the user binary for this rank on the filesystem and open it
  {
    static const char FUNCPTR_SHIM_SUFFIX[] = ".user";

    std::string src{ampi_binary_path};
    src += FUNCPTR_SHIM_SUFFIX;

    std::string dst{src};
    dst += '.';
    dst += std::to_string(rank_count++);
    const char * dststr = dst.c_str();

    if (access(dststr, R_OK) != 0)
      fs_copy(src.c_str(), dststr);

    myexe = dlopen(dststr, RTLD_NOW|RTLD_LOCAL);
  }

  if (myexe == nullptr)
  {
    CkError("dlopen error: %s\n", dlerror());
    CkAbort("Could not open fsglobals user program!");
  }

  return AMPI_FuncPtr_Loader(myexe, argc, argv);
}
