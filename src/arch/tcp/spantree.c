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
 * Revision 2.4  1995-10-19 18:18:24  jyelon
 * added "converse.h"
 *
 * Revision 2.3  1995/09/29  09:50:07  jyelon
 * CmiGet-->CmiDeliver, added protos, etc.
 *
 * Revision 2.2  1995/09/20  15:57:05  sanjeev
 * put void before CmiSpanTreeChildren
 *
 * Revision 2.1  1995/06/12  22:49:59  jyelon
 * *** empty log message ***
 *
 * Revision 1.2  1995/04/13  05:51:12  narain
 * Mc -> Cmi
 *
 * Revision 1.1  1994/11/03  17:37:04  brunner
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";

#include <converse.h>

/**************************
**
** Spanning Tree Functions
**
**************************/

/*
** These are identical to the ipsc2 spanning tree functions.
*/

#define MAXSPAN    4          /* The maximum permitted span on 
				 each node of the spanning tree */

CmiSpanTreeInit() {}


int
CmiSpanTreeParent(node) int node; {
    if (node == 0)
         return -1;
    else 
	return ((node - 1) / MAXSPAN);   /* integer division */
}


int
CmiSpanTreeRoot() { return 0; }


void CmiSpanTreeChildren(node, children) int node, *children; {
    int i;

    for (i = 1; i <= MAXSPAN ; i++)
	if (MAXSPAN * node + i < CmiNumPe())
	     children[i-1] = node * MAXSPAN + i;
	else children[i-1] = -1;
}


int
CmiNumSpanTreeChildren(node) int node; {
    if ((node + 1) * MAXSPAN < CmiNumPe()) return MAXSPAN;
    else if (node * MAXSPAN + 1 >= CmiNumPe()) return 0;
    else return ((CmiNumPe() - 1) - node * MAXSPAN);
}


CmiSendToSpanTreeLeaves(size, msg) int size; char * msg; {
    int node;

    for (node = (CmiNumPe() - 2) / MAXSPAN;   /* integer division */
	 node < CmiNumPe(); node++)
	CmiSyncSendFn(node,size,msg);
}



PrintSpanTree() {
    int i,j;
    int children[MAXSPAN];
    for (i = 0; i < CmiNumPe(); i++) {
	CmiPrintf("node: %d, parent: %d, numchildren: %d, children: ",
		 i, CmiSpanTreeParent(i), CmiNumSpanTreeChildren(i));
	CmiSpanTreeChildren(i, children);
	for (j = 0; j < CmiNumSpanTreeChildren(i); j++)
	     CmiPrintf("%d ", children[j]);
	CmiPrintf("\n");
    }
}
