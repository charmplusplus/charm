#include "charm++.h"
#include "idl.h"

CIHandle ciThisHandle(void *obj)
{
  CIHandle *pHandle = (CIHandle *) ((char *) obj - sizeof(CIHandle));
  return *pHandle;
}

#include "idl.bot.h"
