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


