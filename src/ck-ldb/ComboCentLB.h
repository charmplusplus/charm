/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef CENTRALCOMBOLB_H
#define CENTRALCOMBOLB_H

#include "CentralLB.h"
#include "ComboCentLB.decl.h"

void CreateComboCentLB();

/// for backward compatibility
typedef LBMigrateMsg  CLBMigrateMsg;

class ComboCentLB : public CBase_ComboCentLB
{
public:
  ComboCentLB(const CkLBOptions &);
  ComboCentLB(CkMigrateMessage *m):CBase_ComboCentLB(m) {}

protected:
  virtual bool QueryBalanceNow(int) { return true; };  
  virtual void work(LDStats* stats);

private:  
//  CProxy_CentralLB thisProxy;
  CkVec<CentralLB *>  clbs;
};

#endif /* CENTRALCOMBOLB_H */

/*@}*/


