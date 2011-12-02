#ifndef _MIGRATION_H
#define _MIGRATION_H

#include "migration.decl.h"
#include "megatest.h"

class mig_Element : public CBase_mig_Element
{
 public:
  mig_Element();
  mig_Element(CkMigrateMessage *msg){}
  void done(void);
  void arrive(void);
  void pup(PUP::er &p);
 private:
  int origPE, index, numDone, sum;
};

#endif
