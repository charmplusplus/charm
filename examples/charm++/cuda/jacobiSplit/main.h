#ifndef __MAIN_H__
#define __MAIN_H__

#include "main.decl.h"
#include "jacobi.decl.h"
#include "jacobi_config.h"


/* readonly */ extern CProxy_Main mainProxy;
/* readonly */ extern CProxy_Jacobi jacobiProxy;


class Main : public CBase_Main {

  ////////////////////////////////////////////////////////////////////////////////////////////////
  // Member Variables
  private:
    CProxy_Jacobi jArray;
    int iterationCount;
    double startTime;
    double endTime;

    float partialMaxError[REPORT_MAX_ERROR_BUFFER_DEPTH];
    int checkInCount[REPORT_MAX_ERROR_BUFFER_DEPTH];

    int createdCheckIn_count;

    // STATS
    int reportMaxError_resendCount;

  ////////////////////////////////////////////////////////////////////////////////////////////////
  // Constructor(s) / Destructor
  public:
    Main(CkArgMsg *msg);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  // Member Functions
  public:
    void createdCheckIn();
    void maxErrorReductionClient(CkReductionMsg *msg);
    void reportMaxError(float val, int iter);
};


#endif //__MAIN_H__
