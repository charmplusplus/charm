#include <converse.h>

#if CMK_LBDB_ON

/*
 * This C++ file contains the Charm stub functions
 */

#include "LBDatabase.h"
#include "LBDatabase.def.h"

CkGroupID lbdb;

void CreateLBDatabase()
{
  lbdb = CProxy_LBDatabase::ckNew();
  CkPrintf("New database created\n");
}

#endif // CMK_LBDB_ON
