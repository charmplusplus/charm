#include "Array1D.h"
#include "ArrayElement.h"

ArrayElement::ArrayElement(ArrayElementCreateMessage *msg)
{
  numElements = msg->numElements;
  arrayChareID = msg->arrayID;
  arrayGroupID = msg->groupID;
  thisArray = msg->arrayPtr;
  thisIndex = msg->index;
  //  CPrintf("ArrayElement:%d Creating element %d of %d,%x,%x\n",
  //	  CMyPe(),thisIndex,numElements,
  //	  thisArray,CLocalBranch(Array1D,arrayGroupID));

  delete msg;
}

ArrayElement::ArrayElement(ArrayElementMigrateMessage *msg)
{
  numElements = msg->numElements;
  arrayChareID = msg->arrayID;
  arrayGroupID = msg->groupID;
  thisArray = msg->arrayPtr;
  thisIndex = msg->index;
#if 0
  CPrintf("ArrayElement:%d Receiving migrated element %d\n",
	  CMyPe(),thisIndex,numElements,
	  thisArray,CLocalBranch(Array1D,arrayGroupID));
#endif
  delete msg;
}

void ArrayElement::finishConstruction(void)
{
  //  CPrintf("Finish Constructor registering %d,%d\n",thisIndex,thishandle);
  thisArray->RecvElementID(thisIndex, this, thishandle);
}

void ArrayElement::finishMigration(void)
{
#if 0
  CPrintf("Finish Migration registering %d,%d\n",thisIndex,thishandle);
#endif
  thisArray->RecvMigratedElementID(thisIndex, this, thishandle);
}

void ArrayElement::migrate(int where)
{
#if 0
  CPrintf("Migrating element %d to %d\n",thisIndex,where);
#endif
  if (where != CMyPe())
    thisArray->migrateMe(thisIndex,where);
#if 0
  else 
    CPrintf("PE %d I won't migrating element %d to myself\n",
	    where,thisIndex);
#endif

}

int ArrayElement::packsize(void)
{ 
  CPrintf("ArrayElement::packsize not defined!\n");
  return 0;
}

void ArrayElement::pack(void *pack)
{ 
  CPrintf("ArrayElement::pack not defined!\n");
}

void ArrayElement::exit(ArrayElementExitMessage *msg)
{
  delete msg;
#if 0
  CPrintf("ArrayElement::exit exiting %d\n",thisIndex);
#endif
  ChareExit();
}

#include "ArrayElement.bot.h"





