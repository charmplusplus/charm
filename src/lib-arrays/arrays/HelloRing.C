#include "HelloRing.h"

HelloRing::HelloRing(ArrayElementCreateMessage *msg) : ArrayElement(msg)
{
#if 0
   CPrintf("PE %d creating HelloRing %d %d\n",CMyPe(),thisIndex,thishandle);
#endif
   srand(CMyPe());

   finishConstruction();
}

HelloRing::HelloRing(ArrayElementMigrateMessage *msg) : ArrayElement(msg)
{
#if 0
   CPrintf("PE %d migrating HelloRing %d %d\n",CMyPe(),thisIndex,thishandle);
#endif

   finishMigration();
}

void HelloRing::hello(HelloMessage *msg)
{
  CPrintf("Hello from PE %d, index %d of %d\n",CMyPe(),thisIndex,numElements);
  
  const int maxHops = 100;

  msg->hop++;
  if (msg->hop < maxHops) {
    CPrintf("PE %d Index %d Sending to index %d\n",
	    CMyPe(),thisIndex,(thisIndex+1) % numElements);
    thisArray->send(msg, (thisIndex+1) % numElements,
		    EntryIndex(HelloRing,hello,HelloMessage));
    int migrate_to = rand() % CNumPes();
    CPrintf("PE %d Index %d Migrating to %d\n",CMyPe(),thisIndex,migrate_to);
    this->migrate(migrate_to);
  }
  else CharmExit();

}

int HelloRing::packsize(void)
{
#if 0
  CPrintf("HelloRing packsize = %d\n",sizeof(HelloRing));
#endif
  return sizeof(HelloRing);
}

void HelloRing::pack(void *buf)
{
  HelloRing *hbuf = (HelloRing *)buf;
#if 0
  CPrintf("HelloRing packing into %d\n",buf);
#endif
  hbuf->x = x;
}

#include "HelloRing.bot.h"
