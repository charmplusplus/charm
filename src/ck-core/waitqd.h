//////////////////////////////////////////////////
//
//  waitqd.h
//
//  
//
//  Author: Michael Lang
//  Created: 7/15/99
//
//////////////////////////////////////////////////

#ifndef _WAITQD_H
#define _WAITQD_H

#include "charm++.h"
#include "waitqd.decl.h"

extern "C" void CkWaitQD(void);

class waitqd_QDChare : public Chare {
 private:
   int waitStarted;
   void *threadList;
 public:
   waitqd_QDChare(CkArgMsg *ckam);
   void waitQD(void);
   void onQD(CkQdMsg *ckqm);
};

#endif
