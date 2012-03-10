#include "ck.h"

#undef CMK_G95

#ifdef CMK_G95
extern "C" void g95_runtime_start(int argc, char *argv[]);
extern "C" void g95_runtime_stop();
#endif

#ifndef __BIGSIM__
int main(int argc, char **argv)
{
  int stack_top=0;
  memory_stack_top = &stack_top;
#ifdef CMK_G95
  g95_runtime_start(argc, argv);
#endif
  ConverseInit(argc, argv, (CmiStartFn) _initCharm, 0, 0);
#ifdef CMK_G95
    // FIXME:  not right place to call, but not calling it does not quite hurt
  g95_runtime_stop();
#endif
  return 0;
}
#endif
