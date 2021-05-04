#include "migration.h"

void migration_init(void)
{
  const int numElements = 10 + (CkNumPes() * 2);
  if(CkNumPes() < 2) {
    CkError("migration: requires at least 2 processors.\n");
    megatest_finish();
  } else
    CProxy_mig_Element::ckNew(numElements);
}

void migration_moduleinit(void){}

mig_Element::mig_Element()
{
  origPE = -1;
  sum = 0;
  index = thisIndex;
  numDone = 0;
  CProxy_mig_Element self(thisArrayID);
  self[thisIndex].arrive();
}

void mig_Element::pup(PUP::er &p)
{
  p(origPE);
  p(sum);
  p(numDone);
  p(index);
}

void 
mig_Element::arrive(void)
{
  if (thisIndex != index) 
    CkAbort("migration: chare-data corrupted!\n");
  CProxy_mig_Element self(thisArrayID);
  if(CkMyPe() == origPE) {
    if(sum != ((CkNumPes()+1)*CkNumPes())/2)
      CkAbort("migrate: Element did not migrate to all the processors!\n");
    self[0].checkDisabledMigration();
  } else {
    if(origPE==(-1)) origPE = CkMyPe();
    sum += CkMyPe() + 1;
    int nextPE = (CkMyPe()+1)%CkNumPes();
    migrateMe(nextPE);
  }  
}

void
mig_Element::ckJustMigrated()
{
  CProxy_mig_Element self(thisArrayID);
  self[thisIndex].arrive();
}

void
mig_Element::checkDisabledMigration(void)
{
  CProxy_mig_Element self(thisArrayID);
  _isAnytimeMigration = false;
  int currPE = CkMyPe();
  int nextPE = (CkMyPe()+1)%CkNumPes();
  migrateMe(nextPE);
  CkWaitQD();
  if(CkyPe() != currPE) {
    CkAbort("migrate: Element migrated when anytime migration disabled!\n");
  }
  self[0].done();
}

void mig_Element::done(void) {
  numDone++;
  if(numDone == ckGetArraySize()) {
    _isAnytimeMigration = true;
    megatest_finish();
  }
}

MEGATEST_REGISTER_TEST(migration,"jackie",1)
#include "migration.def.h"
