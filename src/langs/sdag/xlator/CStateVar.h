/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _CStateVar_H_
#define _CStateVar_H_

#include "CString.h"

class CStateVar {
  public:
    CString *type;
    CString *name;
    CStateVar(CString *t, CString *n) : type(t), name(n) {}
};

#endif
