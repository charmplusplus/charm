/*
  Generic converse configuration-flags header.
*/
#ifndef __UIUC_CHARM_CONV_CONFIG_H
#define __UIUC_CHARM_CONV_CONFIG_H

/* Include the machine.c configuration header 
  (e.g., charm/src/arch/net/conv-common.h )
*/ 
#include "conv-common.h"

/* Include the actual platform header 
  (e.g., charm/src/arch/net-linux/conv-mach.h )
*/
#include "conv-mach.h"

/* Include the automatically determined options */
#include "conv-autoconfig.h"

/* Include the build-time options */
#include "conv-mach-opt.h"


/* Fix various invariants (these symbols should probably 
  be eliminated entirely) */
#define CMK_LBDB_OFF (!CMK_LBDB_ON)

#ifndef CMK_USE_HP_MAIN_FIX
# define CMK_USE_HP_MAIN_FIX 0
#endif
#define CMK_DONT_USE_HP_MAIN_FIX  (!CMK_USE_HP_MAIN_FIX)

#ifndef CMK_STACK_GROWDOWN
# define CMK_STACK_GROWDOWN 1
#endif

#endif
