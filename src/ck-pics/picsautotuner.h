#ifndef __AUTOTUNER__H__
#define __AUTOTUNER__H__

#include "TraceAutoTuner.decl.h"
#include "register.h"

class AutoTuneInit : public Chare {
public:
  AutoTuneInit(CkArgMsg *m);
  AutoTuneInit(CkMigrateMessage*m):Chare(m){}
};

class AutoTunerBOC : public CBase_AutoTunerBOC {

public:
  AutoTunerBOC(CkMigrateMessage *m) ;
  AutoTunerBOC() ;
  void registerParameter(ParameterMsg*);
  double getTunedParameter(const char* name, int *valid);
  void tune(vector<IntDoubleMap> perfProblem, int numOfSets);
  void applyTuneResults();
  void updateTP(TunnableParameterUpdateMsg *msg);
  void updateMe(TunnableParameterUpdateMsg *msg);
  void broadcastResults(TunnableParameterUpdateMsg *msg, int index, int configSize, bool bcast);
  //pack the control point values on this PE and broadcast to all other groups
  void packAndBroadcast();
  void printCPToFile(FILE *fp);
  void printCPNameToFile(FILE *fp);
};

class TunnableParameterUpdateMsg : public CMessage_TunnableParameterUpdateMsg {
public:
  int size;
  int source;
  char *names;
  double *values;
};

class ParameterMsg : public CMessage_ParameterMsg {
public:
  char    name[30];
  enum    TP_DATATYPE datatype;
  double  defaultValue;
  double  currentValue;
  double  minValue;
  double  maxValue;
  double  bestValue;
  double  moveUnit;
  int     moveOP;
  int     effect;
  int     effectDirection;
  int     strategy;	
  int     uniqueSet;
  int     chareIdx;
  int     userChareIdx;

  ParameterMsg( const char* n, enum TP_DATATYPE t, double defaultV, double minV, double maxV, double mu, int eff, int dir, int op, int strat, int uSet)
  {
    strcpy(name, n);
    datatype = t;
    bestValue = currentValue = defaultValue = defaultV;
    minValue = minV;
    maxValue = maxV;
    moveUnit = mu;
    moveOP = op;
    effect = eff;
    effectDirection = dir;
    strategy = strat;
    uniqueSet = uSet;
    chareIdx = -1;
  }
  void setChare(char *chareName) {
    chareIdx = CkGetChareIdx(chareName);
  }
  void setUserChareIdx(int _id) {
    userChareIdx = _id;
  }
  void setMoveUnit(double mu) { moveUnit = mu;    }
  void setEffect ( int eff) { effect = eff; }
  void setMoveOP ( int op) { moveOP = op; }
  void setStrategy(int i) { strategy = i;}
};


#endif
