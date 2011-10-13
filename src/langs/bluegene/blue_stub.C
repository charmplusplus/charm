#include "charm++.h"

#ifdef __BIGSIM__

#if  CMK_BIGSIM_THREAD
void BgEmulatorInit(int argc, char **argv)
{
  BgSetWorkerThreadStart(_initCharm);
}
void BgNodeStart(int argc, char **argv) {}
#else
void BgEmulatorInit(int argc, char **argv) {}
void BgNodeStart(int argc, char **argv)
{
  _initCharm(argc, argv);
}
#endif

#else

/* some compiler (power5 xlc) does not like empty object file when making shared lib */
void BgEmulatorInit(int argc, char **argv)
{
}
void BgNodeStart(int argc, char **argv)
{
}

#endif

