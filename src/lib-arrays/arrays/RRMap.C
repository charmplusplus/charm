#include "RRMap.h"


RRMap::RRMap(ArrayMapCreateMessage *msg) : ArrayMap(msg)
{
  CPrintf("PE %d creating RRMap for %d elements\n",CMyPe(),numElements);

  finishConstruction();
}

RRMap::~RRMap()
{
  CPrintf("Bye from RRMap\n");
}

int RRMap::procNum(int element)
{
  return ((element+1) % CNumPes());
}

#include "RRMap.bot.h"
