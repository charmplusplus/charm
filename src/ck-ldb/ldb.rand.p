/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile$
 *	$Author$	$Locker$		$State$
 *	$Revision$	$Date$
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 ***************************************************************************
 * REVISION HISTORY:
 *
 * $Log$
 * Revision 2.4  1997-12-22 21:57:23  jyelon
 * Changed LDB initialization scheme.
 *
 * Revision 2.3  1995/10/27 21:35:54  jyelon
 * changed NumPe --> NumPes
 *
 * Revision 2.2  1995/07/19  22:15:20  jyelon
 * *** empty log message ***
 *
 * Revision 2.1  1995/07/06  22:40:05  narain
 * LdbBocNum, interface to newseed fns.
 *
 * Revision 2.0  1995/06/29  21:19:36  narain
 * *** empty log message ***
 *
 ***************************************************************************/
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * The RAND Load Balancing Strategy* * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#define LDB_ELEMENT void

module ldb {
#include "ldb.h"

export_to_C CldModuleInit()
{
  CldCommonInit();
}

export_to_C CldGetLdbSize()
{
  return 0;
}

export_to_C CldCreateBoc()
{
}

export_to_C CldFillLdb(destPe, ldb)
    int destPe;
    void *ldb;
{
}

export_to_C CldStripLdb(ldb)
    void *ldb;
{
}

export_to_C CldNewSeedFromNet(msgst,ldb,sendfn,queueing,priolen,prioptr)
     void *msgst, *ldb;
     void (*sendfn)();
     unsigned int queueing, priolen, *prioptr;
{
    CsdEnqueueGeneral(msgst, queueing, priolen, prioptr);
}

export_to_C CldNewSeedFromLocal(msgst,ldb,sendfn,queueing,priolen,prioptr)
     void *msgst, *ldb;
     void (*sendfn)();
     unsigned int queueing, priolen, *prioptr;
{
  int pe = rand() % CkNumPes();
  if (pe == CkMyPe())
    CsdEnqueueGeneral(msgst, queueing, priolen, prioptr);
  else
    (*sendfn)(msgst, pe);
}

export_to_C CldProcessMsg(msgPtr, localdataPtr)
void *msgPtr, *localdataPtr;
{
  CkFreeMsg(msgPtr);
}

export_to_C CldProcessorIdle()
{
}

export_to_C CldPeriodicCheckInit()
{
}

}


