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
 * Revision 1.2  1995-10-19 18:18:24  jyelon
 * added "converse.h"
 *
 * Revision 1.1  1995/10/13  16:08:54  gursoy
 * Initial revision
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

void CmiSpanTreeChild(node, children)
int node, *children;
{
    int i;

    for (i = 1; i <= MAXSPAN ; i++)
	if (MAXSPAN * node + i < CmiNumPe())
	     children[i-1] = node * MAXSPAN + i;
	else children[i-1] = -1;
}


int CmiNumSpanTreeChildren(node)
int node;
{
    if ((node + 1) * MAXSPAN < CmiNumPe())
         return MAXSPAN;
    else if (node * MAXSPAN + 1 >= CmiNumPe())
	 return 0;
    else return ((CmiNumPe() - 1) - node * MAXSPAN);
}

CmiSendToSpanTreeLeaves(size, msg)
int size;
char * msg;
{
    int node;

    for (node = (CmiNumPe() - 2) / MAXSPAN;   /* integer division */
	 node < CmiNumPe(); node++)
        CmiSyncSend(node, size, msg);
}

