#ifndef BASELB_H
#define BASELB_H

#include "LBDatabase.h"

class BaseLB: public Group
{
protected:
  LBDatabase *theLbdb;
  LDBarrierReceiver receiver;
public:
  BaseLB() ;
  void unregister(); 
};

#endif
