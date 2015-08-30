#include <stdlib.h>
#include <time.h>
#include "charm++.h"
#include "picsdefs.h"
#include "picsdefscpp.h"
#include "TraceAutoTuner.decl.h"
#include "picstunableparameter.h"
#include "picsautotuner.h"
#include "picsautoperf.h"

extern int  SteeringMethod;
CkpvExtern(SavedPerfDatabase*, perfDatabase);
CkpvExtern(CkCallback, callBackAutoPerfDone);
CkpvDeclare(ParameterDatabase*, allParametersDatabase);
CProxy_AutoTunerBOC autoTunerProxy;

void _createTraceautoTuner(char **argv) { }

void AutoTunerBOC::registerParameter(ParameterMsg* msg)
{
  TunableParameter *mykb = new TunableParameter( msg->name, msg->datatype, msg->defaultValue, 
    msg->minValue, msg->maxValue, msg->moveUnit, msg->moveOP, msg->effect, msg->effectDirection, msg->strategy, msg->uniqueSet, msg->chareIdx, msg->userChareIdx);
  CkpvAccess(allParametersDatabase)->insert(mykb->name, mykb);
  delete msg;
}

double AutoTunerBOC::getTunedParameter(const char* name, int *valid)
{
  double value = CkpvAccess(allParametersDatabase)->getCurrentValue(name, valid, 1);
  return value;
}

void AutoTunerBOC::printCPToFile(FILE *fp) {
  CkpvAccess(allParametersDatabase)->printToFile(fp);
}

void AutoTunerBOC::printCPNameToFile(FILE *fp) {
  CkpvAccess(allParametersDatabase)->printNameToFile(fp);
}

void AutoTunerBOC::packAndBroadcast(){
  TunnableParameterUpdateMsg *msg = CkpvAccess(allParametersDatabase)->packMe();
  autoTunerProxy.updateMe(msg);
}

void AutoTunerBOC::tune(vector<IntDoubleMap> solutions, int numOfSets)
{
  int tune_strategy;
  tune_strategy = TS_LINEAR;
  CkpvAccess(allParametersDatabase)->tuneParameters(tune_strategy, solutions, numOfSets);
  autoPerfProxy[CkpvAccess(myInterGroupParent)].tuneDone();
}

void AutoTunerBOC::applyTuneResults() { }

void AutoTunerBOC::broadcastResults(TunnableParameterUpdateMsg *msg, int index, int configSize, bool bcast) {

  if(treeBranchFactor < 0 || bcast) {
    autoTunerProxy.updateTP(msg);   
  }
  else    //group root, multicast to children
  {
    //one PE makes all decisions,and assign different config to differnt groups   
    int groupWithSameConfig = numGroups/configSize;
    for(int i=0; i<groupWithSameConfig; i++){
      int pe = (groupWithSameConfig * index + i) * treeGroupSize ;
      TunnableParameterUpdateMsg *copymsg = (TunnableParameterUpdateMsg*)CkCopyMsg((void**)&msg);
      autoTunerProxy[pe].updateTP(copymsg);
    }
    delete msg;
  }
}

void AutoTunerBOC::updateTP(TunnableParameterUpdateMsg *msg)
{
  //only for scalable tree
  if(treeBranchFactor > 0) {
    for(int i=0; i<CkpvAccess(numChildren); i++)
    {
      int idInTree = CkMyPe()%treeGroupSize;
      int treeGroupID = CkMyPe()/treeGroupSize;
      int start = treeGroupID * treeGroupSize;
      int child = idInTree*treeBranchFactor+1+i+start;
      TunnableParameterUpdateMsg *copymsg = (TunnableParameterUpdateMsg*)CkCopyMsg((void**)&msg);
      autoTunerProxy[child].updateTP(copymsg);
    }
  }
  CkpvAccess(allParametersDatabase)->updateValues(msg); //self
}

void AutoTunerBOC::updateMe(TunnableParameterUpdateMsg *msg)
{
  CkpvAccess(allParametersDatabase)->updateValues(msg); //self
}

AutoTunerBOC::AutoTunerBOC() {
#if 0
  // which compression algorithm to use
  TunableParameter *tp_compress_algo = new TunableParameter( "RTS_compression_algo", TP_INT, 1, 
    0, 4, 1, OP_ADD, PICS_EFF_UNKNOWN, 1, TS_EXHAUSTIVE, 1, -1, -1);
  CkpvAccess(allParametersDatabase)->insert("RTS_compression_algo", tp_compress_algo);

  TunableParameter *cldb_neighbor_ldb_freq= new TunableParameter("_charm_pics_neighbor_ldb_freq", TP_INT, 20, 20, 20, 2, OP_MUL, PICS_EFF_LDBFREQUENCY, -1, TS_SIMPLE, 2, -1, -1);
  CkpvAccess(allParametersDatabase)->insert("_charm_pics_neighbor_ldb_freq", cldb_neighbor_ldb_freq);
  TunableParameter *is_ldb_step= new TunableParameter("isLdbStep", TP_INT, 1, 0, 1, 2, OP_MUL, PICS_EFF_LDBFREQUENCY, -1, TS_SIMPLE, 1, -1, -1);
  CkpvAccess(allParametersDatabase)->insert("isLdbStep", is_ldb_step);

  char names[200];
  for(int i=0; i<_entryTable.size(); i++)
  {
    EntryInfo *myentry = _entryTable[i];
    if(myentry->appWork)
    {
      sprintf(names,"RTS_COMPRESS_EP_%d_%s", i, myentry->name);
      TunableParameter *mytp = new TunableParameter( names, TP_INT, 0, 
        0, 4, 1, OP_ADD, PICS_EFF_MESSAGESIZE|PICS_EFF_COMPRESSION, 1, TS_EXHAUSTIVE, 1, -1, -1);
      CkpvAccess(allParametersDatabase)->insert(mytp->name, mytp); 
    }
  }
  for(int i=0; i<_entryTable.size(); i++)
  {
    EntryInfo *myentry = _entryTable[i];
    if(myentry->mirror) {
      TunableParameter *mytp_mirror = new TunableParameter( "MIRROR_NUM_SAME", TP_INT, 0, 
        0, 8, 1, OP_ADD, PICS_EFF_MESSAGESIZE, 1, TS_PERF_GUIDE, 1, -1, -1);
      CkpvAccess(allParametersDatabase)->insert(mytp_mirror->name, mytp_mirror);
      break;
    }
  }
  srand (time(NULL));
#endif
}

AutoTunerBOC::AutoTunerBOC(CkMigrateMessage *m) : CBase_AutoTunerBOC(m) { }

void initTunerPerCore()
{
  CkpvInitialize(ParameterDatabase*, allParametersDatabase);
  CkpvAccess(allParametersDatabase) = new ParameterDatabase();
  SteeringMethod = PERF_ANALYSIS_STEER;
}

AutoTuneInit::AutoTuneInit(CkArgMsg *m)
{
  autoTunerProxy = CProxy_AutoTunerBOC::ckNew();
}

#include "TraceAutoTuner.def.h"
