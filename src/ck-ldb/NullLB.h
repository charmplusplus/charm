/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef __CK_NULLLB_H
#define __CK_NULLLB_H

#include <LBDatabase.h>
#include "NullLB.decl.h"

void CreateNullLB(void);

class NullLB : public Group
{
  CProxy_NullLB thisproxy;
  LBDatabase *theLbdb;
  void init(void);
public:
  NullLB() {init();}
  NullLB(CkMigrateMessage *m) {init();}

  static void staticAtSync(void*);
  void AtSync(void); // Everything is at the PE barrier

  void migrationsDone(void);
};

#endif /* def(thisHeader) */
