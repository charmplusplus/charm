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
 * Revision 1.1  1995-11-15 18:17:37  gursoy
 * Initial revision
 *
 * Revision 2.6  1995/11/02  23:10:06  jyelon
 * Fixed CmiSendToSpanTreeLeaves
 *
 * Revision 2.5  1995/10/19  18:20:46  jyelon
 * Changed 'csend' --> 'CmiSyncSend'
 *
 * Revision 2.4  1995/10/19  18:18:24  jyelon
 * added "converse.h"
 *
 * Revision 2.3  1995/09/29  09:50:07  jyelon
 * CmiGet-->CmiDeliver, added protos, etc.
 *
 * Revision 2.2  1995/09/20  15:57:05  sanjeev
 * put void before CmiSpanTreeChildren
 *
 * Revision 2.1  1995/09/20  15:12:09  sanjeev
 * CmiSpanTreeChild -> CmiSpanTreeChildren
 *
 * Revision 2.0  1995/06/23  20:00:01  gursoy
 * Initial Revision
 *
 * Revision 1.1  1995/06/23  19:57:30  gursoy
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";

#include <converse.h>

/* This file contains all the spanning tree functions */

#define MAXSPAN    4          /* The maximum permitted span on 
				 each node of the spanning tree */
#define MAXNODES   256
#define MAXCUBEDIM 8          /* log_2 (MAXNODES) */


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
	if (MAXSPAN * node + i < numnodes())
	     children[i-1] = node * MAXSPAN + i;
	else children[i-1] = -1;
}


int CmiNumSpanTreeChildren(node)
int node;
{
    if ((node + 1) * MAXSPAN < numnodes())
         return MAXSPAN;
    else if (node * MAXSPAN + 1 >= numnodes())
	 return 0;
    else return ((numnodes() - 1) - node * MAXSPAN);
}

CmiSendToSpanTreeLeaves(size, msg)
int size;
char * msg;
{
    int node;

    for (node = (numnodes() - 2) / MAXSPAN;   /* integer division */
	 node < numnodes(); node++)
        CmiSyncSend(node, size, msg);
}
