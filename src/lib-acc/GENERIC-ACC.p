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
 * Revision 1.1  1995-06-13 10:45:22  jyelon
 * Initial revision
 *
 * Revision 1.1  1995/06/13  10:06:34  jyelon
 * Initial revision
 *
 * Revision 1.1  1994/10/17  15:52:37  brunner
 * Initial revision
 *
 ***************************************************************************/

#include "GENERIC-ACC.int"

module GENERIC_MODULE_NAME {

message {
    	GENERIC_DATATYPE data;
} MSG;

accumulator {

	MSG *msg;

	MSG *initfn(x)
	MSG *x;
	{
		msg = (MSG *) CkAllocMsg(MSG);
		msg->data = (GENERIC_DATATYPE) 0;
		return(msg);
	}

	addfn (x)
	GENERIC_DATATYPE x;
	{
		msg->data +=x;
	}

	combinefn (y)
	MSG *y;
	{
		msg->data += y->data;
	}
}  GENERIC_ACC_NAME;


Create()
{
	MSG *msg;

	msg = (MSG *) CkAllocMsg(MSG);
	return(CreateAcc(GENERIC_ACC_NAME, msg));
}

Add(id, i)
int id; 
GENERIC_DATATYPE i;
{	
    	Accumulate(id, addfn(i));
}

Collect(id, ep, cid)
int id;
int ep;
ChareIDType *cid;
{
  	CollectValue(id, ep, cid);
}

}
