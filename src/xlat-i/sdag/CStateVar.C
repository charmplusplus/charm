#include "CStateVar.h"
#include "xi-Parameter.h"

namespace xi {

CStateVar::CStateVar(int v, const char *t, int np, const char *n, XStr *r, const char *a, int m)
  : isVoid(v)
  , numPtrs(np)
  , byRef(r)
  , isMsg(m)
  , declaredRef(NULL)
  , byConst(false)
  , isCounter(false)
  , isSpeculator(false)
  , isBgParentLog(false)
{
  if (t != NULL) { type = new XStr(t); }
  else {type = NULL;}
  if (n != NULL) { name = new XStr(n); }
  else { name = NULL; }
  if (a != NULL) {arrayLength = new XStr(a); }
  else { arrayLength = NULL; }
}

CStateVar::CStateVar(ParamList *pl)
  : isVoid(0)
  , type(new XStr(pl->param->type->isMessage() ? *(pl->param->type->deref()) : *(pl->param->getType())))
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

EncapState::EncapState(Entry* entry, std::list<CStateVar*>& vars)
  : entry(entry)
  , type(0)
  , name(0)
  , isMessage(false)
  , isForall(false)
  , isBgParentLog(false)
  , vars(vars)
{ }

}   // namespace xi
