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
    XStr *type;
    XStr *name;
    CStateVar(XStr *t, XStr *n) : type(t), name(n) {}
};

#endif
