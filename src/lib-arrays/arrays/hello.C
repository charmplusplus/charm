#include <stdio.h>
#include "arraydefs.h"

#include "Array1D.h"
#include "RRMap.h"
#include "HelloRing.h"

#include "hello.top.h"

class main : public chare_object
{
public:
  Array1D *array;
  GroupIDType arrayGroup;

  main(int argc, char *argv[])
  {
    const int numElements = 50;

    CPrintf("Array hello\n");

    arrayGroup =
      Array1D::CreateArray(numElements,
	ChareIndex(RRMap),
	ConstructorIndex(RRMap,ArrayMapCreateMessage),
	ChareIndex(HelloRing),
	ConstructorIndex(HelloRing,ArrayElementCreateMessage),
	ConstructorIndex(HelloRing,ArrayElementMigrateMessage));

    CStartQuiescence(GetEntryPtr(main,quiescenceDetected,QuiescenceMessage),
		     thishandle);
    CPrintf("Leaving main\n");
  };

  void quiescenceDetected(QuiescenceMessage *msg)
  {
    CPrintf("Quiescence detected\n");

    ArrayMessage *sendMsg = new (MessageIndex(HelloMessage)) HelloMessage;
    array=CLocalBranch(Array1D,arrayGroup);
    
    CPrintf("Array = %d,%x\n",arrayGroup,array);
    array->send(sendMsg, 0, EntryIndex(HelloRing,hello,HelloMessage));
    delete msg;
  };
};


#include "hello.bot.h"
