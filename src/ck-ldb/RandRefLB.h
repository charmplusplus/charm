/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _RANDREFLB_H_
#define _RANDREFLB_H_

#include "CentralLB.h"
#include "RandCentLB.h"
#include "RandRefLB.decl.h"

void CreateRandRefLB();

class RandRefLB : public RandCentLB {
public:
  RandRefLB();
  RandRefLB(CkMigrateMessage *m):RandCentLB(m) {}
  void work(CentralLB::LDStats* stats, int count);
};

#endif /* _RANDCENTLB_H_ */


/*@}*/
