#ifndef __TUNABLEPARAMETER__H__
#define __TUNABLEPARAMETER__H__

#include <string.h>
#include <vector>
#include <set>
#include <map>
#include <list>
#include <utility>
#include <charm++.h>
#include <ck.h>
#include <ckarray.h>
#include "picsdefs.h"
#include "picsdefscpp.h"
#include "picsautotunerAPI.h"
#include "picsautotuner.h"

#include "fitCurveTuner.h"

#define  MAX_NAME_LENS      200   

using namespace std;

extern std::map<int,int> chareMap;

class TunableParameter{
public:
  int     numNewTarget;
  int     stableTestCnt;
  int     warmCnt;
  char    name[MAX_NAME_LENS];
  enum    TP_DATATYPE datatype;

  double minValue;
  double maxValue;
  double currentValue;
  double tunedValue;
  double bestValue;
  double defaultValue;
  double previousValue;
  int    tuneDirection;
  //what is the minimum unit to change
  double moveUnit;
  int    moveOP;
  int    effect;
  int    effectDirection;     
  double effectScale;
  int    chareIdx; 
  int    strategy;	//how to search optimal for this parameter 
  int    numofTurns;  //use this to avoid useless/repeat search

  double accelerator;
  double velocity;
  vector<double> valueSet;
  double searchLower;
  double searchUpper;

  //history value
  std::list<double> historyValueList;

  //flag to tell whether this parameter is already tuned
  int isTuned;
  short isTouched; //whether in this phase (or step) this control point is used
  double ldbDiff;

  TunableParameter(const char* n, enum TP_DATATYPE t, double defaultV, double minV, double maxV, int uniqueSet)
  {
    strcpy(name, n);
    datatype = t;
    tunedValue = currentValue = defaultValue = defaultV;
    minValue = minV;
    maxValue = maxV;
    tuneDirection = 1;
    moveUnit = 1;
    moveOP = OP_ADD;
    effect = PICS_EFF_UNKNOWN;
    effectDirection = 1;
    accelerator = 1;
    velocity = 1;
    numofTurns = 0; 
    strategy = TS_SIMPLE;
    stableTestCnt = 0;
    warmCnt = 0;
    isTuned = 0;
    numNewTarget = uniqueSet;
    isTouched = 1;
    ldbDiff = 0;
    chareIdx = -1;
  }

  TunableParameter(const char* n, enum TP_DATATYPE t, double defaultV, double minV, double maxV, double mu, int op, int eff, int effDir, int strat, int uniqueSet, int _chareIdx, int userChareIdx)
  {
    strcpy(name, n);
    datatype = t;
    tunedValue = currentValue = defaultValue = defaultV;
    minValue = minV;
    maxValue = maxV;
    moveUnit = mu;
    moveOP =op;
    effect = eff;
    effectDirection = effDir;
    strategy = strat;
    tuneDirection = 1;
    accelerator = 1;
    velocity = 1;
    numofTurns = 0; 
    stableTestCnt = 0;
    warmCnt = 0;
    isTuned = 0;
    numNewTarget = uniqueSet;
    isTouched = 1;
    ldbDiff = 0;
    chareIdx = _chareIdx;
    if(chareIdx>0){
      chareMap[chareIdx] = userChareIdx;
    }
  }

  void setNumNewTarget(int n) {
    numNewTarget = n;
  }
  int getNumNewTarget() {
    return numNewTarget;
  }

  int setTunedValue(double f);
  int setNewValue(double f);
  void feedback(double f1, double f2);

  int configSize() {
    return valueSet.size();
  }

  double getConfig(int i) {
    return valueSet[i];
  }

  void reverseDirection()
  {
    numofTurns++; 
    tuneDirection = - tuneDirection;
  }

  void saveValue(double f)
  {
    if(historyValueList.size() >= 10)
    {
      historyValueList.pop_front();
    }
    historyValueList.push_back(f);
  }

  double getCurrentValue(){
    return currentValue;
  }

  double getTunedValue() {
    return tunedValue;
  }

  void setCurrentIsTuned() {
    currentValue = tunedValue;
  }
  void setTouched(short t) {
    isTouched = t;
  }

  short getTouched () {
    return isTouched;
  }

  void saveToBest() { 
    bestValue = currentValue;
  }

  void printInfo() {
    CkPrintf(" Name %s \n", name);
  }

  void useBest()  {
    saveValue(currentValue);
    tunedValue = bestValue;
  }

  void adjustValue(double &f);
  void move();
  void randMove();
  void exhaustiveMove(double change);
  void simpleMove(double change);
  void linearMove(double change);
  void exponentialMove(double change);
  void randomMove(double change);
  void biSearch(double change);
  void perfModelOpt(double change);
  void tune(double perfmetrics, int effect);
  void tuneLDB(double perfmetrics, int effect, int ldbStep);
  void reset();
  void printMe()
  {
    CkPrintf("===== using CP   %s  effect is %d value    %.1f   =====>  %.1f  \n", name, effect, previousValue, currentValue);
  }
};

struct cmp_str
{
  bool operator()(char const *a, char const *b)
  {
    return strcmp(a, b) < 0;
  }
};

typedef std::map<const char*, TunableParameter*, cmp_str> parameter_map;

class ParameterDatabase {
private:
  parameter_map  parametersList;
  vector< vector< TunableParameter*> > effectsList;

public:
  ParameterDatabase() {
    effectsList.resize(PICS_NUM_EFFECTS);
  }

  void insert(const char *name, TunableParameter* tp);

  double getCurrentValue(const char *name, int *valid, short isTouched) {
    parameter_map::iterator iter = parametersList.find(name); 
    if(iter != parametersList.end())
    {
      *valid = 1; 
      iter->second->setTouched(isTouched);
      iter->second->setCurrentIsTuned();
      return iter->second->getCurrentValue();
    }else
    {
      *valid = 0; 
      return 0;
    }
  }

  void useBestParameters(double metricsChangeRatio, int strategy, int effect)
  {
    for(parameter_map::iterator iter = parametersList.begin(); iter != parametersList.end(); ++iter)
    {
      TunableParameter *mykb  = iter->second;
      mykb->useBest();
    }
  }

  void printAll();
  void learn();
  void tuneParameters(int strategy, vector<IntDoubleMap> &effects, int numOfSets);
  void updateValues(TunnableParameterUpdateMsg *msg);

  void generateCombination(vector<TunableParameter*> &tunedVec, int requiredNum, vector<vector<double > > &configs);
  void DFS(vector<TunableParameter*> &tunedVec, int pos, int requiredNum, vector<double> &current, vector<vector<double> > &result);

  TunnableParameterUpdateMsg* packMe() {
    int size = parametersList.size();
    TunnableParameterUpdateMsg *tpUpdateMsg = new (MAX_NAME_LENS*size, size) TunnableParameterUpdateMsg;  
    tpUpdateMsg->source = CkMyPe();
    tpUpdateMsg->size = size;
    int i=0;
    for(parameter_map::iterator iter = parametersList.begin(); iter != parametersList.end(); ++iter)
    {
      TunableParameter *mykb  = iter->second;
      strcpy(tpUpdateMsg->names + i*MAX_NAME_LENS, mykb->name);
      tpUpdateMsg->values[i] = mykb->getCurrentValue();
      i++;
    }
    return tpUpdateMsg;
  }

  void printToFile(FILE *fp);
  void printNameToFile(FILE *fp);
};

#endif
