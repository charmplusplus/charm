#include <converse.h>
#include "commlib.h"

typedef struct buf{
  int size;
  void * msg;
  struct buf * next;
} Buffer;

//Dimensional Exchange (Hypercube) based router
class BcastRouter : public Router
{
  private:
	comID MyID;
  	int MyPe, NumPes, recvCount, *gpes;
	int PSendCounter, PSendExpected;
	Buffer *MsgBuffer;
	void InitVars();
  public:
	BcastRouter(int, int);
	~BcastRouter();
	void NumDeposits(comID, int);
	void EachToAllMulticast(comID, int , void *, int);
	void EachToManyMulticast(comID, int , void *, int, int *, int);
	void ProcMsg(int, msgstruct **) {;}
	void RecvManyMsg(comID, char *);
	void ProcManyMsg(comID, char *);
	void DummyEP(comID id, int) {;}
	void SetID(comID);
	void SetMap(int *);
};
