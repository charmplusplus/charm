#define LDB_ELEMENT void

module ldb {
#include "ldb.h"

message {
    int dummy;
} DUMMYMSG;

export_to_C CldModuleInit()
{
  CldCommonInit();
}

export_to_C setLdbSize()
{
       CpvAccess(LDB_ELEM_SIZE) = sizeof(LDB_ELEMENT);
}

export_to_C LdbCreateBoc()
{
  DUMMYMSG *msg;
  msg = (DUMMYMSG *)CkAllocMsg(DUMMYMSG);	
  CreateBoc(LDB, LDB@BranchInit, msg);
}



export_to_C LdbFillLDB(ldb)
LDB_ELEMENT *ldb;
{
	BranchCall(CpvAccess(LdbBocNum), LDB@FillLDB(ldb));
}

export_to_C LdbStripLDB(ldb)
LDB_ELEMENT *ldb;
{
	BranchCall(CpvAccess(LdbBocNum), LDB@StripLDB(ldb));
}


export_to_C Ldb_NewMsg_FromNet(msg) 
void *msg;
{
	BranchCall(CpvAccess(LdbBocNum), LDB@NewMsg_FromNet(msg) );
}

export_to_C Ldb_NewMsg_FromLocal(msg)
void *msg;
{
	BranchCall(CpvAccess(LdbBocNum), LDB@NewMsg_FromLocal(msg) );
}

export_to_C LdbProcessMsg(msgPtr, localdataPtr)
void *msgPtr, *localdataPtr;
{
	BranchCall(CpvAccess(LdbBocNum), LDB@ProcessMsg(msgPtr, localdataPtr));
}

export_to_C LdbProcessorIdle()
{
	BranchCall(CpvAccess(LdbBocNum), LDB@ProcessorIdle());
}


export_to_C LdbPeriodicCheckInit()
{
	BranchCall(CpvAccess(LdbBocNum) , LDB@PeriodicCheckInit());
}


BranchOffice LDB {

int	numNeighbours;
int	NumPE;
int 	myPE;
int *neighboursList;
int LdbBoc;

public FillLDB(ldb)
LDB_ELEMENT *ldb;
{
}

public StripLDB(ldb)
LDB_ELEMENT *ldb;
{
}


public NewMsg_FromNet(msg) 
void *msg;
{
    QsEnqUsrMsg(msg);
}

public NewMsg_FromLocal(msg)
void *msg;
{
  int destPE;

  if (!DestFixed_Msg(msg)) {
    destPE = determine_msg_trans(msg);
/*
    TRACE(CkPrintf("[%d] Translation for event=%d, pe=%d, to %d\n",
		   CkMyPe(), GetEnv_event(x), GetEnv_pe(x), 
		   destPE));
*/
  }
  if (destPE == CkMyPe())
    QsEnqUsrMsg(msg);
  else
    SEND_TO(msg, destPE);
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

entry BranchInit : (message DUMMYMSG * dmsg)
{
	int i;
	
	TRACE(CkPrintf("Enter Node LdbInit()\n"));
	CpvAccess(LDB_ELEM_SIZE) = 0;
	LdbBoc = MyBocNum();
	CpvAccess(LdbBocNum) = LdbBoc;
	NumPE = CkNumPes();
	myPE = CkMyPe();
	numNeighbours = CkNumNeighbours(myPE);
	TRACE(CkPrintf("Node LdbInit() Done: NumPE %d, numNeighbours %d\n",
		NumPE, numNeighbours));
}


}

}



