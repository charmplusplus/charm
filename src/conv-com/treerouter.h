/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef TREEROUTER_H
#define TREEROUTER_H
#include <converse.h>
#include "comlib.h"
#include "petable.h"

class TreeRouter : public Router
{
  private:
	PeTable *PeTree;
	int numExpect, *gpes;
 	int *totarray;
  	int MyPe, NumPes, numChildren, recvCount, recvExpected;
	void InitVars();
	void DownStreamMsg(comID id);
	void LocalProcMsg(comID);
	char * SortBufferUp(comID, int, int *);
	char * SortBufferDown(comID, int, int *, int);
  public:
	TreeRouter(int, int);
	~TreeRouter();
	void NumDeposits(comID, int);
	void EachToAllMulticast(comID, int , void *, int);
	void EachToManyMulticast(comID, int , void *, int, int *, int);
	void RecvManyMsg(comID, char *);
	void ProcManyMsg(comID, char *);
	void DummyEP(comID, int );
	void SetMap(int *);
};

#endif
	
