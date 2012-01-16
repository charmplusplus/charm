/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef __CK_NULLLB_H
#define __CK_NULLLB_H

#include <BaseLB.h>
#include "NullLB.decl.h"

/**
 NullLB is inherited from BaseLB. It has all the strategy's API, 
 but doing nothing but resume from sync
 NullLB only is functioning when there is no other strategy created.
*/
class NullLB : public BaseLB
{
public:
  NullLB(const CkLBOptions &opt): BaseLB(opt) 
	{init(); lbname="NullLB";}
  NullLB(CkMigrateMessage *m):BaseLB(m){ }
  ~NullLB();

  static void staticAtSync(void*);
  void AtSync(void); // Everything is at the PE barrier

  void migrationsDone(void);
  void pup(PUP::er &p){ 
    BaseLB::pup(p); 
    if(p.isUnpacking()) init(); 
    lbname="NullLB"; 
  }
private:
  CProxy_NullLB thisProxy;
  void init();
};

#endif /* def(thisHeader) */


/*@}*/
