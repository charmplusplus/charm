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
 * Revision 1.2  1997-07-31 18:50:32  rbrunner
 * New T3D version based on T3E version.  Old files are now in t3d-fm
 *
 * Revision 1.1  1997/07/07 22:57:21  rbrunner
 * T3E port
 *
 * Revision 1.1  1996/05/16 15:59:43  gursoy
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
	if (MAXSPAN * node + i < _num_pes())
	     children[i-1] = node * MAXSPAN + i;
	else children[i-1] = -1;
}


int CmiNumSpanTreeChildren(node)
int node;
{
    if ((node + 1) * MAXSPAN < _num_pes())
         return MAXSPAN;
    else if (node * MAXSPAN + 1 >= _num_pes())
	 return 0;
    else return ((_num_pes() - 1) - node * MAXSPAN);
}

CmiSendToSpanTreeLeaves(size, msg)
int size;
char * msg;
{
    int node;

    for (node = (_num_pes() - 2) / MAXSPAN;   /* integer division */
	 node < _num_pes(); node++)
        CmiSyncSend(node, size, msg);
}
