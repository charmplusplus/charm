#include "Array1D.h"
#include "ArrayMap.h"

ArrayMap::ArrayMap(ArrayMapCreateMessage *msg)
{
  CPrintf("PE %d creating ArrayMap\n",CMyPe());
  arrayChareID = msg->arrayID;
  arrayGroupID = msg->groupID;
  array = CLocalBranch(Array1D, arrayGroupID);
  numElements = msg->numElements;

  delete msg;
}

void ArrayMap::finishConstruction(void)
{
  array->RecvMapID(this, thishandle, thisgroup);
}

#include "ArrayMap.bot.h"
