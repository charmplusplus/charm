/**
 * \addtogroup CkLdb
*/
/*@{*/

/*
 Startup routine for use when you include all the load balancers.
*/
#include <LBDatabase.h>
#include "EveryLB.decl.h"

static void CreateNoLB(void)
{
	/*empty-- let the user create the load balancer*/
}

void initEveryLB(void) {
#if CMK_LBDB_ON
//  LBSetDefaultCreate(CreateNoLB);
#endif
}

#include "EveryLB.def.h"

/*@}*/
