/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _GRIDROUTER_H
#define _GRIDROUTER_H

#include <math.h>
#include <converse.h>
#include "commlib.h"
#include "petable.h"

#define ROWLEN COLLEN
#define RowLen(pe) ColLen(pe)
#define PELISTSIZE ((ROWLEN-1)/sizeof(int)+1)

inline int ColLen(int npes)
{
  int len= (int)sqrt((double)npes);
  if (npes > (len*len)) len++;
  return(len);
}

inline int Expect(int pe, int npes)
{
  int i, len=ColLen(npes);
  for (i=len-1;i>=0;i--) {
  	int myrow=pe/len;
	int toprep=i*len;
	int offset=pe-myrow*len;
	if ((toprep+offset) <= (npes-1)) return(i+1);
  }
  return(len);
}

//Grid based router
class GridRouter : public Router
{
  private:
	PeTable *PeMesh;
	comID MyID;
  	int *onerow, *gpes;
	int MyPe, NumPes, COLLEN;
 	int LPMsgCount, LPMsgExpected;
	int recvExpected, recvCount;
	void InitVars();
	void LocalProcMsg();
  public:
	GridRouter(int, int);
	~GridRouter();
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
