/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include "charm++.h"
#include "idl.h"

CIHandle ciThisHandle(void *obj)
{
  CIHandle *pHandle = (CIHandle *) ((char *) obj - sizeof(CIHandle));
  return *pHandle;
}

#include "idl.def.h"
