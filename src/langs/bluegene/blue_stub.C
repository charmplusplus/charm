#include "charm++.h"

#ifdef __BLUEGENE__

#if  CMK_BLUEGENE_THREAD
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
#endif

