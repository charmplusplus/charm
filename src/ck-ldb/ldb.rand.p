/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile$
 *	$Author$	$Locker$		$State$
 *	$Revision$	$Date$
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 ***************************************************************************
 * REVISION HISTORY:
 *
 * $Log$
 * Revision 2.1  1995-07-06 22:40:05  narain
 * LdbBocNum, interface to newseed fns.
 *
 * Revision 2.0  1995/06/29  21:19:36  narain
 * *** empty log message ***
 *
 ***************************************************************************/
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * The RAND Load Balancing Strategy* * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#define LDB_ELEMENT void

module ldb {
#include "ldb.h"

message {
    int dummy;
} DUMMYMSG;

export_to_C getLdbSize()
{
       return 0;
}

export_to_C LdbCreateBoc()
{
  DUMMYMSG *msg;
  msg = (DUMMYMSG *)CkAllocMsg(DUMMYMSG);	
  CreateBoc(LDB, LDB@BranchInit, msg);
}

export_to_C LdbFillLDB(destPe, ldb)
  int destPe;	
  void *ldb;
{
  BranchCall(ReadValue(LdbBocNum), LDB@FillLDB(destPe, ldb));
}

export_to_C LdbStripLDB(ldb)
     void *ldb;
{
  BranchCall(ReadValue(LdbBocNum), LDB@StripLDB(ldb));
}


export_to_C Ldb_NewSeed_FromNet(msgst, ldb, sendfn) 
     void *msgst, *ldb;	
	void (*sendfn)();	
{
  BranchCall(ReadValue(LdbBocNum), LDB@NewMsg_FromNet(msgst, ldb, sendfn) );
}

export_to_C Ldb_NewSeed_FromLocal(msgst, ldb, sendfn)
     void *msgst, *ldb;	
	void (*sendfn)();	
{
  BranchCall(ReadValue(LdbBocNum), LDB@NewMsg_FromLocal(msgst, ldb, sendfn) );
}

export_to_C LdbProcessMsg(msgPtr, localdataPtr)
void *msgPtr, *localdataPtr;
{
	BranchCall(ReadValue(LdbBocNum), LDB@ProcessMsg(msgPtr, localdataPtr));
}

export_to_C LdbProcessorIdle()
{
	BranchCall(ReadValue(LdbBocNum), LDB@ProcessorIdle());
}


export_to_C LdbPeriodicCheckInit()
{
	BranchCall(ReadValue(LdbBocNum) , LDB@PeriodicCheckInit());
}


BranchOffice LDB {

int   NeedLdbStripMsg;
int	numNeighbours;
int	NumPE;
int 	myPE;
int *neighboursList;
int LdbBoc;

private Strategy(msg, sendfn)
void *msg;
void (*sendfn)();
{
    int pe = rand() % CmiNumPe();
    if (pe == myPE)
         CsdEnqueue(msg);
    else
      (*sendfn)(msg, pe);	
}


entry BranchInit : (message DUMMYMSG * dmsg)
{
	int i;
	
	TRACE(CkPrintf("Enter Node LdbInit()\n"));
	LdbBoc = MyBocNum();
	LdbBocNum = LdbBoc;
	ReadInit(LdbBocNum) ;
	NumPE = CmiNumPe();
	myPE = CmiMyPe();
	numNeighbours = CmiNumNeighbours(myPE);
	TRACE(CkPrintf("Node LdbInit() Done: NumPE %d, numNeighbours %d\n",
		NumPE, numNeighbours));
}


public FillLDB(destPe, ldb)
int destPe;
void *ldb;
{
}

public StripLDB(ldb)
void *ldb;
{
}


public NewMsg_FromNet(msgst, ldb, sendfn) 
     void *msgst, *ldb;	
	void (*sendfn)();	
{
    CsdEnqueue(msgst);
}

public NewMsg_FromLocal( msgst, ldb, sendfn)
	void *msgst, *ldb;	
	void (*sendfn)();	
{
    PrivateCall(Strategy(msgst, sendfn));
}

public ProcessMsg(msgPtr, localdataPtr)
void *msgPtr, *localdataPtr;
{
  CkFreeMsg(msgPtr);		/* After processing the message, free it.*/
}

public ProcessorIdle()
{
}


public PeriodicCheckInit()
{
}


}

}


