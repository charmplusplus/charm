/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _CStateVar_H_
#define _CStateVar_H_

#include "xi-util.h"

class CStateVar {
  public:
    XStr *isconst;
    int isVoid;
    XStr *type1;
    XStr *type2;
    XStr *allPtrs;
    int numPtrs;
    XStr *name;
    XStr *byRef;
    XStr *arrayLength;
    int isMsg;
    CStateVar(XStr *c, int v, XStr *t1, XStr *t2, XStr *p, int np, XStr *n, XStr *r, XStr *a, int m) : isconst(c), isVoid(v), type1(t1), type2(t2), allPtrs(p), numPtrs(np), name(n), byRef(r), arrayLength(a), isMsg(m) {}

};

#endif
