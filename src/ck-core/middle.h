#ifndef _MIDDLE_H_
#define _MIDDLE_H_

#include "conv-config.h"  /* If we don't make sure this is included, we may miss CMK_BIGSIM_CHARM */

#if CMK_BIGSIM_CHARM
# include "middle-blue.h"
  using namespace BGConverse;
#else
# include "middle-conv.h"
  using namespace Converse;
#endif

#endif
