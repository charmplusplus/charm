#ifndef _CStateVar_H_
#define _CStateVar_H_

#include "xi-util.h"
#include "xi-symbol.h"

#include <list>

namespace xi {
  class ParamList;

  struct CStateVar {
    int isVoid;
    XStr *type;
    int numPtrs;
    XStr *name;
    XStr *byRef, *declaredRef;
    bool byConst;
    XStr *arrayLength;
    int isMsg;
    bool isCounter, isSpeculator, isBgParentLog;

    CStateVar(int v, const char *t, int np, const char *n, XStr *r, const char *a, int m);
    CStateVar(ParamList *pl);
  };

  struct EncapState {
    Entry* entry;
    XStr* type;
    XStr* name;
    bool isMessage;
    bool isForall;
    bool isBgParentLog;
    std::list<CStateVar*> vars;

    EncapState(Entry* entry, std::list<CStateVar*>& vars);
  };
}

#endif
