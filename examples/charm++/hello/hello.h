//////////////////////////////////////
//
//  hello.h  
//
//  Declaration of chares in hello
//
//  Author: Michael Lang
//  Date: 6/15/99
//
//////////////////////////////////////

#include "hello.decl.h"

class main : public Chare {
public:
  main(CkMigrateMessage *m) {}
  main(CkArgMsg *m);
};
