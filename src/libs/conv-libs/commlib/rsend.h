/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <converse.h>
#include "commlib.h"

//Dimensional Exchange (Hypercube) based router
class RsendRouter : public Router
{
  private:
	comID MyID;
	int *gpes;
  	int MyPe, NumPes;
	int PSendCounter, PSendExpected;
  public:
	RsendRouter(int, int);
	~RsendRouter();
	void NumDeposits(comID, int);
	void EachToAllMulticast(comID, int , void *, int );
	void EachToManyMulticast(comID, int , void *, int, int *, int);
	void ProcMsg(int, msgstruct **) {;}
	void RecvManyMsg(comID, char *);
	void ProcManyMsg(comID, char *);
	void DummyEP(comID id, int);
	void SetID(comID);
	void SetMap(int *);
};
