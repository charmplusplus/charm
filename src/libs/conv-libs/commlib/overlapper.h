/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _OVERLAPPER_H
#define _OVERLAPPER_H

typedef struct odb {
  int refno;
  int magic;
  comID id;
  struct odb * next;
} OverlapDummyBuffer;

typedef struct orb {
  int refno;
  char * msg;
  comID id;
  struct orb * next;
} OverlapRecvBuffer;

typedef struct obb {
  int more;
  int npe;
  int *pelist;
  int msgsize;
  void * msg;
  comID id;
  struct obb * next;
} OverlapBuffer;

typedef OverlapRecvBuffer OverlapProcBuffer;
class Overlapper {
  private :
	comID MyID;
	int MyPe, NumPes, NoSwitch, gnpes, *gpes;
  	int Active, RecvActive, ActiveRefno;
	int SwitchDecision, MoreDeposits, DeleteFlag;
	OverlapBuffer *OBFirst, *OBLast, *OBFreeList;
	OverlapRecvBuffer *ORFirst, *ORFreeList;
   	OverlapDummyBuffer *ODFirst, *ODFreeList;
	OverlapProcBuffer *OPFirst, *OPFreeList;
	void StartNext();
	void CommAnalyzer(comID);
	void SwitchStrategy();
	void InsertSwitchMsgs(int, int *);
	Router * routerObj;
	
  public :
	Overlapper(comID);
	~Overlapper();
	void NumDeposits(comID, int);
	void EachToAllMulticast(comID id, int size, void *msg);
	void EachToManyMulticast(comID, int , void *, int, int *);
	void NumPSends(comID, int) {;}
	void RecvManyMsg(comID, char *) ;
	void DummyEP(comID, int, int);
	void ProcManyMsg(comID id, char *msg);
	void SetID(comID);
	int MyActiveIndex() {return(ActiveRefno);}
	void Done();
	void GroupMap(int, int *);
	void DeleteInstance();
};
	

#endif
