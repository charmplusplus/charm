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
 * Revision 1.4  1995-10-27 21:45:35  jyelon
 * Changed CmiNumPe --> CmiNumPes
 *
 * Revision 1.3  1995/10/25  19:56:05  jyelon
 * Changed CmiSyncSendFn --> CmiSyncSend
 *
 * Revision 1.2  1995/10/19  18:18:24  jyelon
 * added "converse.h"
 *
 * Revision 1.1  1995/09/30  15:00:00  jyelon
 * Initial revision
 *
 * Revision 2.2  1995/09/29  09:50:07  jyelon
 * CmiGet-->CmiDeliver, added protos, etc.
 *
 * Revision 2.1  1995/09/20  15:11:45  sanjeev
 * CmiSpanTreeChild -> CmiSpanTreeChildren
 *
 * Revision 2.0  1995/07/05  23:37:59  gursoy
 * *** empty log message ***
 *
 *
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";

/* This file contains all the spanning tree functions */
#include <converse.h>


#define MAXSPAN    4          /* The maximum permitted span on 
				 each node of the spanning tree */

CmiSpanTreeInit()
{
}


int CmiSpanTreeParent(node)
int node;
{
    if (node == 0)
         return -1;
    else return ((node - 1) / MAXSPAN);   /* integer division */
}

int CmiSpanTreeRoot()
{
    return 0;
}

void CmiSpanTreeChildren(node, children)
int node, *children;
{
    int i;

    for (i = 1; i <= MAXSPAN ; i++)
	if (MAXSPAN * node + i < CmiNumPes())
	     children[i-1] = node * MAXSPAN + i;
	else children[i-1] = -1;
}


int CmiNumSpanTreeChildren(node)
int node;
{
    if ((node + 1) * MAXSPAN < CmiNumPes())
         return MAXSPAN;
    else if (node * MAXSPAN + 1 >= CmiNumPes())
	 return 0;
    else return ((CmiNumPes() - 1) - node * MAXSPAN);
}

CmiSendToSpanTreeLeaves(size, msg)
int size;
char * msg;
{
    int node;

    for (node = (CmiNumPes() - 2) / MAXSPAN;   /* integer division */
	 node < CmiNumPes(); node++)
        CmiSyncSend(node, size, msg);
}

