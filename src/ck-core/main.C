#include "ck.h"

#ifndef __BIGSIM__
int main(int argc, char **argv)
{
  int stack_top=0;
  memory_stack_top = &stack_top;

  ConverseInit(argc, argv, (CmiStartFn) _initCharm, 0, 0);

  return 0;
}
#endif
