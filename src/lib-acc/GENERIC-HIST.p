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
 * Revision 1.2  1995-06-13 11:30:13  jyelon
 * *** empty log message ***
 *
 * Revision 1.1  1995/06/13  10:45:22  jyelon
 * Initial revision
 *
 * Revision 1.1  1995/06/13  10:06:34  jyelon
 * Initial revision
 *
 * Revision 1.1  1994/10/17  15:52:37  brunner
 * Initial revision
 *
 ***************************************************************************/

#include HEAD

module GENERIC_MODULE_NAME {

message {
	int size;
    	varSize GENERIC_DATATYPE data[];
} MSG;

message {
	int data;
} DataMsg;

accumulator {

	MSG *msg;

	MSG *initfn (x)
	DataMsg *x;
	{
		int i;
		int sizes[1];

		sizes[0] = x->data;
		msg = (MSG *) CkAllocMsg(MSG, sizes);
		msg->size = x->data;
		for (i=0; i<msg->size; i++)
			msg->data[i] = (GENERIC_DATATYPE) 0;
		return(msg);
	}
	
	addfn (x, y)
	int x;
	GENERIC_DATATYPE y;
	{
		msg->data[x] += y;
	}

	combinefn (y)
	MSG *y;
	{
		int 	i;
		for (i=0; i<msg->size; i++)
			msg->data[i] += y->data[i];
	}

} GENERIC_ACC_NAME;



Create(n)
int n;
{
	DataMsg *msg;

	msg = (DataMsg *) CkAllocMsg(DataMsg);
	msg->data = n;
	return(CreateAcc(GENERIC_ACC_NAME, msg));
}

Add(id, i, j)
int id; 
int i;
GENERIC_DATATYPE j;
{	
    	Accumulate(id, addfn(i, j));
}

Collect(id, ep, cid)
int id;
int ep;
ChareIDType *cid;
{
  	CollectValue(id, ep, cid);
}

}
