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

#ifndef __CK_NULLLB_H
#define __CK_NULLLB_H

#include <BaseLB.h>
#include "NullLB.decl.h"

void CreateNullLB(void);

/**
 NullLB is inherited from BaseLB. It has all the strategy's API, 
 but doing nothing but resume from sync
 NullLB only is functioning when there is no other strategy created.
*/
class NullLB : public CBase_NullLB
{
  void init(void);
public:
  NullLB() {init(); lbname="NullLB";}
  NullLB(CkMigrateMessage *m):CBase_NullLB(m) {init();}
  ~NullLB();

  static void staticAtSync(void*);
  void AtSync(void); // Everything is at the PE barrier

  void migrationsDone(void);
};

#endif /* def(thisHeader) */


/*@}*/
