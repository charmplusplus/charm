/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _D3GRIDROUTER_H
#define _D3GRIDROUTER_H

#include <math.h>
#include <converse.h>
#include "commlib.h"
#include "petable.h"

//3DGrid based router
class D3GridRouter : public Router
{
  private:
	PeTable *PeGrid;
	comID MyID;
  	int *oneplane, *zline, *gpes;
	int MyPe, NumPes, COLLEN, nPlanes;
 	int LPMsgCount, LPMsgExpected;
	int recvExpected[2], recvCount[2];
        int routerStage;
	void InitVars();
	void LocalProcMsg();
  public:
	D3GridRouter(int, int);
	~D3GridRouter();
	void NumDeposits(comID, int);
	void EachToAllMulticast(comID, int , void *, int);
	void EachToManyMulticast(comID, int , void *, int, int *, int);
	void RecvManyMsg(comID, char *);
	void ProcManyMsg(comID, char *);
	void DummyEP(comID id, int);
	void ProcMsg(int, msgstruct **) {;}
	void SetID(comID);
	void SetMap(int *);
};

#endif
