#include "ck.h"

int main(int argc, char **argv)
{
  ConverseInit(argc, argv, (CmiStartFn) _initCharm, 0, 0);
  return 0;
}
