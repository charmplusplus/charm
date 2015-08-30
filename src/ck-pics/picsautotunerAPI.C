/*
 * =====================================================================================
 *
 *       Filename:  autoTuneAPI.C
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  07/21/2013 09:11:28 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include "picsdefs.h"
#include "picsdefscpp.h"
#include "picsautotunerAPI.h"
#include "picsautotuner.h"
#include "picstunableparameter.h"

extern CProxy_AutoTunerBOC autoTunerProxy;
CkpvExtern(ParameterDatabase*, allParametersDatabase);

void PICS_registerLocalTunableParameterFields(char *name, TP_DATATYPE dt, double dv, double minV, double  maxV, double mU, int moveOP, int eff, int effDir, int stp)
{
  TunableParameter *mykb = new TunableParameter(name, dt, dv, minV, maxV, mU, moveOP, eff, effDir, stp, 1, -1, -1);
  CkpvAccess(allParametersDatabase)->insert(mykb->name, mykb);
}

void PICS_registerTunableParameterFields(char *name, TP_DATATYPE dt, double dv, double minV, double  maxV, double mU, int moveOP,  int eff, int effDir, int stp, int uniqueSet)
{
  ParameterMsg *param = new ParameterMsg(name, dt, dv, minV, maxV, mU, moveOP, eff, effDir, stp, uniqueSet);  
  autoTunerProxy.registerParameter(param);
}

void PICS_registerTunableParameterFieldsWithChare(char *name, TP_DATATYPE dt, double dv, double minV, double  maxV, double mU, int moveOP,  int eff, int effDir, int stp, int uniqueSet, int _id, char *chareName)
{
  ParameterMsg *param = new ParameterMsg(name, dt, dv, minV, maxV, mU, moveOP, eff, effDir, stp, uniqueSet);  
  param->setChare(chareName);
  param->setUserChareIdx(_id);
  autoTunerProxy.registerParameter(param);
}

void PICS_registerTunableParameter( ControllableParameter *tp)
{
  ParameterMsg *param = new ParameterMsg(tp->name, tp->datatype, tp->defaultValue, tp->minValue, tp->maxValue, tp->moveUnit, tp->moveOP, tp->effect, tp->effectDirection, tp->strategy, tp->uniqueSets);  
  autoTunerProxy.registerParameter(param);
}

void PICS_registerLocalTunableParameter(ParameterMsg *param)
{
  autoTunerProxy.ckLocalBranch()->registerParameter(param);
}

double PICS_getTunedParameter(const char *name, int *valid)
{
  *valid = 0;
  double value;
  if(!autoTunerProxy.ckGetGroupID().isZero())
    value = autoTunerProxy.ckLocalBranch()->getTunedParameter(name, valid);
  else
    value = 0;
  return value;
}

