#ifndef _CStateVar_H_
#define _CStateVar_H_

#include "xi-util.h"

namespace xi {

class CStateVar {
  public:
    int isVoid;
    XStr *type;
    int numPtrs;
    XStr *name;
    XStr *byRef;
    XStr *arrayLength;
    int isMsg;
    CStateVar(int v, const char *t, int np, const char *n, XStr *r, const char *a, int m) : isVoid(v), numPtrs(np),  byRef(r), isMsg(m)
 	{ 
	  if (t != NULL) { type = new XStr(t); } 
	  else {type = NULL;}
	  if (n != NULL) { name = new XStr(n); }
	  else { name = NULL; }
	  if (a != NULL) {arrayLength = new XStr(a); }
	  else { arrayLength = NULL; }
	}
};

}

#endif
