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

    CStateVar(int v, const char *t, int np, const char *n, XStr *r, const char *a, int m)
      : isVoid(v)
      , numPtrs(np)
      , byRef(r)
      , isMsg(m)
      , declaredRef(NULL)
      , byConst(false)
      , isCounter(false)
      , isSpeculator(false)
      , isBgParentLog(false) {
      if (t != NULL) { type = new XStr(t); }
      else {type = NULL;}
      if (n != NULL) { name = new XStr(n); }
      else { name = NULL; }
      if (a != NULL) {arrayLength = new XStr(a); }
      else { arrayLength = NULL; }
    }

    CStateVar(ParamList *pl)
      : isVoid(0)
      , type(new XStr(*(pl->param->getType())))
      , numPtrs(0)
      , name(new XStr(pl->getGivenName()))
      , byRef(pl->isReference() ? new XStr("&") : NULL)
      , declaredRef(pl->declaredReference() ? new XStr("&") : NULL)
      , byConst(pl->isConst())
      , arrayLength(pl->isArray() ? new XStr(pl->getArrayLen()) : NULL)
      , isMsg(pl->isMessage())
      , isCounter(false)
      , isSpeculator(false)
      , isBgParentLog(false)
    { }
  };

  struct EncapState {
    Entry* entry;
    XStr* type;
    XStr* name;
    std::list<CStateVar*> vars;

    EncapState(Entry* entry, std::list<CStateVar*>& vars)
      : entry(entry)
      , type(0)
      , name(0)
      , vars(vars) { }
  };

}

#endif
