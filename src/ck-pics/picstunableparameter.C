#include <stdlib.h>
#include <algorithm>
#include "picsdefs.h"
#include "picstunableparameter.h"
#include "picsautoperf.h"
#include "fitCurveTuner.h"
#include "picstreenode.h"
#include "picsdefs.h"

extern  CProxy_AutoTunerBOC autoTunerProxy;
extern  CProxy_TraceAutoPerfBOC autoPerfProxy;

int SteeringMethod;

CkpvExtern(SavedPerfDatabase*, perfDatabase);
CkpvExtern(double, timeForLdb);
CkpvExtern(double, timeBeforeLdb);
CkpvExtern(double, currentTimeStep);
CkpvExtern(int, cntAfterLdb);

std::map<int,int> chareMap;

void TunableParameter::reset()
{
  velocity = 1;
  numofTurns = 0;
  isTuned = 0;
  stableTestCnt = 0;
}

void TunableParameter::move()
{
  double ratio = autoPerfProxy.ckLocalBranch()->getCurrentBestRatio();
  double newValue ;

  previousValue = currentValue;
  if(velocity < 1)
    velocity = 1;

  if(numofTurns == 3){
    numofTurns = 2;
    isTuned = 0;
  }
  double tmpValue = currentValue;
  for(int i=0; i<numNewTarget; i++){
    switch(moveOP)
    {
    case OP_ADD:
      newValue = tmpValue + velocity * moveUnit * tuneDirection; 
      break;
    case OP_MUL:
      if(tuneDirection > 0)
        newValue = tmpValue * velocity * moveUnit; 
      else
        newValue = tmpValue * (1.0/(moveUnit*velocity)); 
      break;
    default:
      break;
    }
    if(newValue == tmpValue)
      break;
    adjustValue(newValue);  //adjust value to the proper range
    valueSet.push_back(newValue);
    tmpValue = newValue;
  }
}

void TunableParameter::adjustValue(double &f){
  if (f > maxValue)
  {
    f = maxValue;
    reverseDirection();
  }
  else if (f< minValue)
  {
      f = minValue;
      reverseDirection();
  }
}

void TunableParameter::randMove()
{
  double v01 = ((double)rand()/(RAND_MAX));
  double newValue = v01 * (maxValue-minValue) + minValue ;
  adjustValue(newValue);  //adjust value to the proper range
  valueSet.push_back(newValue);
}

void TunableParameter::exhaustiveMove(double metricsChangeRatio)
{
  move();
}

int TunableParameter::setTunedValue(double f)
{
  previousValue = currentValue;
  feedback(currentValue, f);
  tunedValue = f;
  return tunedValue;
}

int TunableParameter::setNewValue(double f)
{
  return 0;
}

void TunableParameter::feedback(double f1, double f2)
{
  if(f1 == f2) 
    return ;
  int len1 = 16;
  int ptr = 16;
  char indexStr[10];
  int index;
  if( memcmp(name, "RTS_COMPRESS_EP_", len1) == 0 )
  {
    while(name[ptr] != '_')
    {
      ptr++;
    }
    memcpy(indexStr, name+len1, ptr-len1);
    index = atoi(indexStr);
    _entryTable[index]->compressAlgoIdx = f2; 
  }
}

void TunableParameter::simpleMove(double metricsChangeRatio)
{
  move();
}

void TunableParameter::linearMove(double metricsChangeRatio)
{
  move();
}

void TunableParameter::exponentialMove(double metricsChangeRatio)
{
  move();
}


void TunableParameter::randomMove(double metricsChangeRatio)
{
}

//it is tricky how to use bisearch to speed up 
void TunableParameter::biSearch(double metricsChangeRatio)
{
}

//optimization based on performance model
void TunableParameter::perfModelOpt(double metricsChangeRatio)
{
#if USE_GSL 
  int i;
  int n; 
  int minDataset;
  double *rawx;
  double *rawy;
  switch( strategy)
  {
  case PERF_LINEAR:
    minDataset = 2;
    break;
  case PERF_QUADRIC:
    minDataset = 3;
    break;
  case PERF_CUBIC:
    minDataset = 4;
    break;
  case PERF_QUARTIC:
    minDataset = 5;
    break;
  default:
    CmiAbort("This is not implemented yet\n");
  }
  if(n<5)
  {
    move();
    return;
  }
  rawx = new double[n*sizeof(double)];
  rawy = new double[n*sizeof(double)];
  i=0;
  for(std::list<double>::iterator it=historyValueList.begin(); it != historyValueList.end(); it++, i++)
  {
    rawx[i] = *it;
  }
  CkpvAccess(perfDatabase)->getAllTimeSteps(rawy, n);
  previousValue = currentValue;
  vector< vector<double> > minX;
  minX.resize(1);
  FittingCurve::quadraticFitting(1, n, rawx, rawy, minX);
  minX.clear();

  FittingCurve::cubicFitting(1, n, rawx, rawy, minX);

  minX.clear();
  FittingCurve::quarticFitting(1, n, rawx, rawy, minX );
  if(minX.size() > 0)
  {
    setNewValue(minX[0][0]);
  }

  delete[] rawx;
  delete[] rawy;
#endif
}


void TunableParameter::tuneLDB(double metricsChangeRatio, int eff, int ldbStep)
{
  if(ldbStep == 0)
  {
   ldbDiff = -CkpvAccess(timeForLdb);     
   tunedValue = 0;
  }else
  {
    double prevSave = ldbDiff/ldbStep;
    ldbDiff += ( CkpvAccess(timeBeforeLdb) - CkpvAccess(currentTimeStep));
    if(ldbDiff/(ldbStep+1) > prevSave)
    {
      tunedValue = 0;
    }
    else
      tunedValue = 1;
  }
  valueSet.push_back(tunedValue);
}

void TunableParameter::tune(double metricsChangeRatio, int eff)
{
  int oldTuneDirection = tuneDirection;
  saveValue(currentValue);
  valueSet.clear();
  // repeated search for three times; choose best configuration first
  if(numofTurns == 8 && (currentValue != bestValue ||  (metricsChangeRatio>0.95 && metricsChangeRatio < 1.05))) { 
    valueSet.push_back(bestValue);
    isTuned = 1;
  }
  else
  {
    if(eff == effect)
    {
      tuneDirection = effectDirection;
    }else if (eff == -effect)
    {
      tuneDirection = -effectDirection;
    }else if(eff == PICS_EFF_UNKNOWN || strategy == TS_PERF_GUIDE)
    {
      if(metricsChangeRatio > 1.010 && previousValue != currentValue && strategy != TS_EXHAUSTIVE) 
      {
        reverseDirection();
      }
    }
    switch(strategy)
    {
    case  TS_LINEAR:
      if(oldTuneDirection == tuneDirection)
        velocity++;
      else
        velocity--;
      break;
    case TS_EXPONENTIAL:
      if(oldTuneDirection == tuneDirection)
        velocity *= 2;
      else
        velocity /= 2;
      break;

    case TS_EXHAUSTIVE:
    case TS_SIMPLE: 
      move();
      break;
    case TS_RANDOM:
      randMove();
      break;
    case TS_BISEARCH:
    case PERF_LINEAR:
    case PERF_QUADRIC:
      perfModelOpt(metricsChangeRatio);
      break;
    default:
      CkAbort("Not implement strategy, please switch to others\n");
      break;
    }
  }
  CkPrintf("PE %d  tuning %s  eff %d effect %d  ratio %f from %f ---> %f [%f %f] \n", CkMyPe(), name, eff, effect, metricsChangeRatio, historyValueList.back(), valueSet[0], minValue, maxValue );
}

void  ParameterDatabase::insert(const char *name, TunableParameter* tp)
{
  parametersList[name] = tp;
  effectsList[tp->effect].push_back(tp);
  if(!CkMyPe())
    tp->printMe();
}

void ParameterDatabase::printAll()
{
  parameter_map::iterator iter;
  for(iter=parametersList.begin(); iter!= parametersList.end(); iter++)
  {
    TunableParameter *mykb  = iter->second;
    printf(" Configuration %s---- %.1f \n", iter->first, mykb->getCurrentValue()); 
  }
}

void ParameterDatabase::updateValues(TunnableParameterUpdateMsg *msg)
{
  int size = msg->size;
  TunableParameter *mykb;
  char name[MAX_NAME_LENS];
  for(int i=0; i< size; i++)
  {
    strcpy(name, msg->names+i*MAX_NAME_LENS);
    mykb = parametersList[name];
    mykb->setTunedValue(msg->values[i]);
  }
  delete msg;
}

void ParameterDatabase::learn() {
  //compare all the performance data of this step with previous step 
  //check which condition goes away and then check what control point is tuned
  //map the condition with control point; experimental feature

  PerfData *summaryData = (PerfData*)((CkpvAccess(summaryPerfDatabase)->getData(0))->getData());
  PerfData *prevSummaryData = (PerfData*)((CkpvAccess(summaryPerfDatabase)->getData(-1))->getData());

  if(summaryData == NULL || prevSummaryData == NULL)
    return;

  std::vector<Condition*> removedConditions;
  std::vector<Condition*> addedConditions;
  for(int j=0; j<CkpvAccess(numOfPhases)*PERIOD_PERF; j++)
  {
    //phase to phase comparison 
    //std::vector<Condition*>& conditionsSet1 = prevSummaryData->getPerfProblems();
    //std::vector<Condition*>& conditionsSet2 = summaryData->getPerfProblems();

    std::vector<Condition*> conditionsSet1;
    std::vector<Condition*> conditionsSet2;
    for(std::vector<Condition*>::iterator iter=conditionsSet1.begin(); iter!= conditionsSet1.end(); iter++)
    {
      std::vector<Condition*>::iterator iterFind = std::find(conditionsSet2.begin(), conditionsSet2.end(), *iter);
      if(iterFind == conditionsSet2.end())    //condition iter is removed, condition exists in previous not in current
      {
        removedConditions.push_back(*iterFind); 
      }
    }

    for(std::vector<Condition*>::iterator iter=conditionsSet2.begin(); iter!= conditionsSet2.end(); iter++)
    {
      std::vector<Condition*>::iterator iterFind = std::find(conditionsSet1.begin(), conditionsSet1.end(), *iter);
      if(iterFind == conditionsSet1.end())    //condition iter is removed, condition exists in current not in previous 
      {
        addedConditions.push_back(*iterFind); 
      }
    }

    // removedConditions is a set of conditions which are removed due to the control point change in this step
    // // map the effects with these conditions
    summaryData++;
  }
  CkpvAccess(learnTree)->addNodes();
  //learnTree.add();
  //check which control point is modified and what conditions go away, derive the new rules based on the above 
}

void ParameterDatabase::tuneParameters( int strategy, vector<IntDoubleMap>  &effects, int numOfSet)
{
  double metricsChangeRatio;
  CkReductionMsg *msg1 = CkpvAccess(summaryPerfDatabase)->getData(0);
  CkReductionMsg *msg2 = CkpvAccess(summaryPerfDatabase)->getData(-1);
  PerfData *summaryData = (PerfData*)(msg1->getData());
  PerfData *prevSummaryData = (PerfData*)(msg2->getData());

  vector<TunableParameter*> myvec;
  int n, i;
  int nV;
  int minDataset;
  double *rawx, *rawy;
  int effect, abseffect;
  int vecsize;

  //goals
  if(CkpvAccess(perfGoal) == BestTimeStep)
  {
    metricsChangeRatio = summaryData->timeStep/prevSummaryData->timeStep;
  }else if (CkpvAccess(perfGoal) == BestUtilPercentage){
    metricsChangeRatio = summaryData->utilPercentage/prevSummaryData->utilPercentage;
  }else if (CkpvAccess(perfGoal) == BestEnergyEfficiency){
  }

  if(autoPerfProxy.ckLocalBranch()->isCurrentBest())   //previous  configuration is best so far, save it
  {
    for(parameter_map::iterator iter = parametersList.begin(); iter != parametersList.end(); ++iter)
    {
      TunableParameter *mykb  = iter->second;
      CkAssert(mykb != NULL);
      mykb->saveToBest();
    }
  }

  for(int i=0; i<effects.size(); i++){
    for(IntDoubleMap::iterator iter=effects[i].begin(); iter!= effects[i].end(); iter++)
    {
      effect = iter->first;
      if(effect == PICS_EFF_PERFGOOD)    
      {
        //no need to tune
        for(parameter_map::iterator iter = parametersList.begin(); iter != parametersList.end(); ++iter)
        {
          TunableParameter *mykb  = iter->second;
          mykb->setTouched(0);
        }
        //broadcast my configuration to all
        TunnableParameterUpdateMsg *msg = packMe();
        autoTunerProxy.updateMe(msg);
        return;
      }
    }
  }


  int numOfTunableCP = 0;
  std::map<TunableParameter*, int> tunableCPs;
  for(int i=0; i<effects.size(); i++){
  for(IntDoubleMap::iterator iter=effects[i].begin(); iter!= effects[i].end(); iter++)
  {
    effect = iter->first;
    abseffect = effect>=0 ? effect : -effect;
    if(abseffect >0 && abseffect < PICS_NUM_EFFECTS-1 )
    {
      //tune this effect
      for(vector<TunableParameter*>::iterator iter = effectsList[abseffect].begin(); iter != effectsList[abseffect].end(); ++iter)
      {
        TunableParameter *mykb  = *iter;
        mykb->printMe();
        if(mykb->strategy != PERF_DIRECT && mykb->getTouched())
        {
          mykb->printMe();
          tunableCPs[mykb] = effect;
        }
    }
  }
  }
  }
  int replicas=1;
  numOfTunableCP = tunableCPs.size();
  if(numOfTunableCP>0)
  {
    replicas = pow(numOfSet, 1.0/numOfTunableCP);
    //CkPrintf(" tunable CP is %d each generate %d configs ldb %d \n", numOfTunableCP, replicas, CkpvAccess(cntAfterLdb));
  }
  else
  {
    //CkPrintf(" No tunable CP \n");
    return;
  }
  myvec.clear();
  for(std::map<TunableParameter*, int>::iterator iter = tunableCPs.begin(); iter!= tunableCPs.end(); iter++){
    TunableParameter *mykb = iter->first;
    effect = iter->second;
    mykb->setNumNewTarget(replicas);
    if(mykb->isTuned ==1)
      mykb->reset();
    if(effect == PICS_EFF_LDBFREQUENCY || effect == -PICS_EFF_LDBFREQUENCY){
      if(CkpvAccess(cntAfterLdb) == 1)
      {
        mykb->tuneLDB(metricsChangeRatio, effect, CkpvAccess(cntAfterLdb));
        mykb->setTouched(0);
        myvec.push_back(mykb);
      }
    }else
    {
      mykb->tune(metricsChangeRatio, effect);
      mykb->setTouched(0);
      myvec.push_back(mykb);
    }
  }

  vecsize = myvec.size();
  if(vecsize == 0)
    return;

  //generate all combinations of different configuration
  vector<vector<double > >  configs;
  generateCombination(myvec, numOfSet, configs);

  bool bcast = false;
  if(configs.size() == 1)
    bcast = true;
  for(int j=0; j<configs.size(); j++)
  {
    TunnableParameterUpdateMsg *tpUpdateMsg = new (MAX_NAME_LENS*vecsize, vecsize) TunnableParameterUpdateMsg;  
    tpUpdateMsg->source = CkMyPe();
    tpUpdateMsg->size = vecsize;
    for(i=0; i<myvec.size(); i++)
    {
      TunableParameter *mykb  = myvec[i];
      strcpy(tpUpdateMsg->names + i*MAX_NAME_LENS, mykb->name);
      tpUpdateMsg->values[i] = configs[j][i];
    }
    autoTunerProxy.ckLocalBranch()->broadcastResults(tpUpdateMsg, j, configs.size(), bcast);
  }
  myvec.clear();

}

void ParameterDatabase::generateCombination(vector<TunableParameter*> &tunedVec, int requiredNum,  vector<vector<double > >  &result) 
{
  vector<double> current;
  DFS(tunedVec, 0, requiredNum, current, result);
}

void ParameterDatabase::DFS( vector<TunableParameter*> &tunedVec, int pos, int requiredNum, vector<double> &current, vector<vector<double> > &result){

  if(result.size() == requiredNum)
    return;
  if(pos == tunedVec.size()) {
    //done here 
    result.push_back(current);
  }else
  {
    for(int i=0; i<tunedVec[pos]->configSize(); i++)
    {
      current.push_back(tunedVec[pos]->getConfig(i));
      DFS(tunedVec, pos+1, requiredNum, current, result);
      current.pop_back();
    }
  }
}

void ParameterDatabase::printToFile(FILE *fp) {
  for(parameter_map::iterator iter = parametersList.begin(); iter != parametersList.end(); ++iter)
  {
    TunableParameter *mykb  = iter->second;
    fprintf(fp, "CP %s   %f  %f\n", mykb->name, mykb->getCurrentValue(), mykb->getTunedValue());
  }
}

void ParameterDatabase::printNameToFile(FILE *fp) {
  fprintf(fp, "CPSIZE %d \n", (int)(parametersList.size()));
  for(parameter_map::iterator iter = parametersList.begin(); iter != parametersList.end(); ++iter)
  {
    TunableParameter *mykb  = iter->second;
    fprintf(fp, "CP %s   \n", mykb->name, mykb->getCurrentValue());
  }
}
