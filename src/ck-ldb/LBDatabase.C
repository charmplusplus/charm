#include <converse.h>


/*
 * This C++ file contains the Charm stub functions
 */

#include "LBDatabase.h"
#include "LBDatabase.def.h"


CkGroupID lbdb;

LBDBInit::LBDBInit(CkArgMsg *m)
{
#if CMK_LBDB_ON
  lbdb = CProxy_LBDatabase::ckNew();
  CkPrintf("[%d] New database created\n",CkMyPe());
#endif
  delete m;
}
