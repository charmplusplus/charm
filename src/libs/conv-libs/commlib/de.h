/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _DE_H
#define _DE_H
#include <converse.h>
#include "commlib.h"
#include "petable.h"

//Dimensional Exchange (Hypercube) based router
class DimexRouter : public Router
{
  private:
 	PeTable *PeHcube;
	comID MyID;
	int *buffer;
  	int* msgnum, InitCounter;
	int *penum,*gpes;
	int **next;
  	int Dim, stage, MyPe, NumPes;
	void InitVars();
	void CreateStageTable(int, int *);
	void LocalProcMsg();
  public:
	DimexRouter(int, int);
	~DimexRouter();
	void NumDeposits(comID, int);
	void EachToAllMulticast(comID , int , void *, int);
	void EachToManyMulticast(comID , int , void *, int, int *, int);
	void ProcMsg(int, msgstruct **) {;}
	void RecvManyMsg(comID, char *);
	void ProcManyMsg(comID, char *);
	void DummyEP(comID id, int);
	void SetID(comID id) { MyID=id;}
	void SetMap(int *);
};
#endif
