/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/**
 * \addtogroup CkPerf
*/
/*@{*/


#ifndef _TRACE_BLUEGENE_H
#define _TRACE_BLUEGENE_H

#include "trace.h"
#include "blue.h"
#include "blue_impl.h"

class TraceBluegene : public Trace {

 private:
    bgTimeLog* currLog;
    FILE* stsfp;
    FILE* pfp;
 public:
    TraceBluegene(char** argv);
    ~TraceBluegene();
    int traceOnPE() { return 1; }
    //   void* userBracketEvent(int e, double bt, double et);
    //    void userBracketEvent(char* name,double bt, double et,char* msg,void** parentLogPtr);
    void getForwardDep(void* log, void** fDepPtr);
    void getForwardDepForAll(void** logs1, void** logs2, int logsize,void* fDepPtr);
    void tlineEnd(void** parentLogPtr);
    void bgBeginExec(char* name,void** parentLogPtr);
    void bgEndExec(void);
    void userBracketEvent(char* name, double bt, double et, void** parentLogPtr);
    void userBracketEvent(char* name, double bt, double et, void** parentLogPtr, CkVec<void*> bgLogList);
    void bgPrint(char* str);
    void traceWriteSts();
    void creatFiles();
    void writePrint(char *, double t);
    void traceClose();
};


#endif

/*@}*/
