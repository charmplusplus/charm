#ifndef _MIDDLE_H_
#define _MIDDLE_H_

#include "charm.h"  /* If we don't make sure this is included, we may miss CMK_BLUEGENE_CHARM */

#if CMK_NAMESPACES_BROKEN
# if CMK_BLUEGENE_CHARM
#  error "BLUEGENE Charm++ cannot be compiled without namespace support"
# else
#  include "middle-conv.h"
# endif
#else
# if CMK_BLUEGENE_CHARM
#  include "middle-blue.h"
   using namespace BGConverse;
# else
#  include "middle-conv.h"
   using namespace Converse;
# endif
#endif

#endif
