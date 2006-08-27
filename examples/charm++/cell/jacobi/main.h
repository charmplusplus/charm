#ifndef __MAIN_H__
#define __MAIN_H__

#include "main.decl.h"
#include "jacobi_shared.h"


/* readonly */ CProxy_Main mainProxy;


class Main : public CBase_Main {

  ////////////////////////////////////////////////////////////////////////////////////////////////
  // Member Variables
  private:
    CProxy_Jacobi jArray;
    int iterationCount;
    double startTime;
    double endTime;

  ////////////////////////////////////////////////////////////////////////////////////////////////
  // Constructor(s) / Destructor
  public:
    Main(CkArgMsg *msg);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  // Member Functions
  public:
    void maxErrorReductionClient(CkReductionMsg *msg);
};


#endif //__MAIN_H__
