#ifndef COMLIB_H
#define COMLIB_H
#include <converse.h>
#include <stdlib.h>

#define NULL 0

enum{BCAST=0,TREE, GRID, HCUBE, RSEND};  
#define MAXNUMMSGS 1000
#define MAXNUMSTRATEGY 100
#define MSGSIZETHRESHOLD 50000

typedef struct {
  int srcpe;
  int ImplType;
  int ImplIndex;
  int SwitchVal;
  int NumMembers;
  CmiGroup grp;
} comID;


typedef struct {
  int msgsize;
  void *msg;
} msgstruct ;

class Router;

/*************** Interface to the User **********************/
void ComlibInit();
comID ComlibInstance(int ImplType, int numParticipants);
void NumDeposits(comID  id, int num);
comID ComlibEstablishGroup(comID id, int npes, int *pes); 
Router * GetStrategyObject(int n, int me, int indx);
void DeleteInstance(comID id);

/* Converse messages */
void EachToAllMulticast(comID  id, int size, void * msg);
void EachToManyMulticast(comID id, int size, void *msg, int npe, int * pelist);

/* Charm++ messages */
void CComlibEachToManyMulticast(comID, int, void *, int, int, int *);

/**************** Declarations for the Developer *************/
CpvExtern(int, RecvHandle);
CpvExtern(int, ProcHandle);
CpvExtern(int, DummyHandle);

void KSendDummyMsg(comID id, int pe, int magic);
int KMyActiveRefno(comID);
void KDone(comID);
void KCsdEnqueue(void *m);

//Base class for routers
class Router 
{
  private:
  public:
	Router() {};
	virtual ~Router() {};
	virtual void NumDeposits(comID, int) 
						{CmiPrintf("Not impl\n");}
	virtual void EachToAllMulticast(comID, int , void *, int) 
						{CmiPrintf("Not impl\n");}
	virtual void EachToManyMulticast(comID, int , void *, int, int *, int) 
						{CmiPrintf("Not impl\n");}
	virtual void RecvManyMsg(comID, char *) {CmiPrintf("Not Impl\n");}
	virtual void ProcManyMsg(comID, char *) {CmiPrintf("Not Impl\n");}
	virtual void DummyEP(comID, int ) 	{CmiPrintf("Base Dummy\n");}
	virtual void SetID(comID) {;}
	virtual void SetMap(int *) {;}
};
typedef Router * (*NEWFN)(int, int);


#endif
	

