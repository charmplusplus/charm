/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include "ck.h"

#ifndef __BLUEGENE__
int main(int argc, char **argv)
{
  ConverseInit(argc, argv, (CmiStartFn) _initCharm, 0, 0);
  return 0;
}
#endif
