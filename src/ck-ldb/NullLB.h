/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef __CK_NULLLB_H
#define __CK_NULLLB_H

#include <BaseLB.h>
#include "NullLB.decl.h"

void CreateNullLB(void);

class NullLB : public CBase_NullLB
{
  void init(void);
public:
  NullLB() {init();}
  NullLB(CkMigrateMessage *m) {init();}
  ~NullLB();

  static void staticAtSync(void*);
  void AtSync(void); // Everything is at the PE barrier

  void migrationsDone(void);
};

#endif /* def(thisHeader) */
