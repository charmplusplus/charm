/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <converse.h>


/*
 * This C++ file contains the Charm stub functions
 */

#include "LBDatabase.h"
#include "LBDatabase.def.h"


CkGroupID lbdb;

static LBDefaultCreateFn defaultCreate=NULL;
void LBSetDefaultCreate(LBDefaultCreateFn f)
{
	defaultCreate=f;
}


LBDBInit::LBDBInit(CkArgMsg *m)
{
#if CMK_LBDB_ON
  lbdb = CProxy_LBDatabase::ckNew();
  if (defaultCreate) (defaultCreate)();
#endif
  delete m;
}
