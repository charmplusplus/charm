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
 * Revision 2.2  1998-05-04 17:45:22  rbrunner
 * Random numbers different on nodes
 *
 * Revision 2.1  1997/12/30 23:50:24  jyelon
 * Random LDB based on converse --- no charm involved.
 *
 * Revision 2.4  1997/12/22 21:57:23  jyelon
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

#include <converse.h>

void CldModuleInit()
{
  CldCommonInit();
}

int CldGetLdbSize()
{
  return 0;
}

void CldCreateBoc()
{
}

void CldFillLdb(destPe, ldb)
     int destPe;
     void *ldb;
{
}

void CldStripLdb(ldb)
    void *ldb;
{
}

void CldNewSeedFromNet(msgst,ldb,sendfn,queueing,priolen,prioptr)
     void *msgst, *ldb;
     void (*sendfn)();
     unsigned int queueing, priolen, *prioptr;
{
    CsdEnqueueGeneral(msgst, queueing, priolen, prioptr);
}

void CldNewSeedFromLocal(msgst,ldb,sendfn,queueing,priolen,prioptr)
     void *msgst, *ldb;
     void (*sendfn)();
     unsigned int queueing, priolen, *prioptr;
{
  int pe = (rand()+CmiMyPe()) % CmiNumPes();
  if (pe == CmiMyPe())
    CsdEnqueueGeneral(msgst, queueing, priolen, prioptr);
  else
    (*sendfn)(msgst, pe);
}

void CldProcessorIdle()
{
}

void CldPeriodicCheckInit()
{
}


