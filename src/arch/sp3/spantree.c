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
 * Revision 1.1  1997-07-08 22:10:57  milind
 * Added IBM SP3 version. Developed and Tested on ANL machine.
 *
 * Revision 2.8  1995/10/27 21:45:35  jyelon
 * Changed CmiNumPe --> CmiNumPes
 *
 * Revision 2.7  1995/10/19  18:18:24  jyelon
 * added "converse.h"
 *
 * Revision 2.6  1995/10/02  19:28:29  knauff
 * Fixed the call to CmiAsyncSendFn in CmiSendToSpanTreeLeaves, so I
 * don't need the '#include machine.h' after all.
 *
 * Revision 2.5  1995/10/02  19:22:34  knauff
 * Added '#include machine.h' so PROCESS_PID doesn't barf.
 *
 * Revision 2.4  1995/09/29  09:50:07  jyelon
 * CmiGet-->CmiDeliver, added protos, etc.
 *
 * Revision 2.3  1995/09/20  15:57:05  sanjeev
 * put void before CmiSpanTreeChildren
 *
 * Revision 2.2  1995/09/08  02:38:48  gursoy
 * Cmi_dim is no more exported from machine.c. now thru a function call
 *
 * Revision 2.1  1995/07/17  17:47:17  knauff
 * Added '#include "converse.h"'
 *
 * Revision 2.0  1995/07/10  22:12:53  knauff
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";

/* This file contains all the spanning tree functions */
#include <converse.h>

#define MAXSPAN    4          /* The maximum permitted span on 
				 each node of the spanning tree */
#define MAXNODES   1024
#define MAXCUBEDIM  10         /* log_2 (MAXNODES) */


typedef struct spantreearray {
    int noofchildren;
    int parent;
    int *children;
} SpanTreeArray;

static SpanTreeArray *SpanArray;
static int *NodeStore;   /* used to store the nodes in the spanning 
		       tree in breadth first order */
static int numnodes;

CmiSpanTreeInit()
{
    int i, j;
    int visited[MAXNODES];
    int next, currentnode;
    int neighbours[MAXCUBEDIM];
    int dim;

    numnodes = CmiNumPes();
    dim      = CmiNumNeighbours(0);
    SpanArray = (SpanTreeArray *)CmiAlloc(sizeof(SpanTreeArray) * numnodes);
    NodeStore = (int *) CmiAlloc(sizeof(int) * numnodes);
    visited[0] = 1;
    NodeStore[0] = 0;  /* the root of the spanning tree */
    SpanArray[0].parent = -1;  /* no parent */

    for (i = 1; i < numnodes; i++)
        visited[i] = 0;

    for (next = 1, i = 0; i < numnodes; i++)
    {
	currentnode = NodeStore[i];
	CmiGetNodeNeighbours(currentnode, neighbours);
	SpanArray[currentnode].noofchildren = 0;
	for (j = 0; j < dim && 
	            SpanArray[currentnode].noofchildren < MAXSPAN; j++)
	{
	    if (!visited[neighbours[j]])
	    {
		NodeStore[next + SpanArray[currentnode].noofchildren] = 
								neighbours[j];
		SpanArray[currentnode].noofchildren++;
		SpanArray[neighbours[j]].parent = currentnode;
		visited[neighbours[j]] = 1;
		
	    }
	}
	if (SpanArray[currentnode].noofchildren != 0)
	{
	    SpanArray[currentnode].children = &NodeStore[next];
	    next += SpanArray[currentnode].noofchildren;
	}
    }

    for (i = 0; i < numnodes; i++)  /* check */
	if (!visited[i])
	   CmiError("node %d not part of spanning tree: initialization error!\n",i);
}


int CmiSpanTreeRoot()
{
    return 0;  /* node 0 is always the root of the spanning tree on the 
		  hypercubes */
}


int CmiSpanTreeParent(node)
int node;
{
    return SpanArray[node].parent;
}


void CmiSpanTreeChildren(node, children)
int node, *children;
{
    int i;

    for( i = 0; i < SpanArray[node].noofchildren; i++)
         children[i] = SpanArray[node].children[i];
}


int CmiNumSpanTreeChildren(node)
int node;
{
    return SpanArray[node].noofchildren;
}




PrintSpanTree()
{
    int i,j;

    for (i = 0; i < numnodes; i++)
    {
	CmiPrintf("node: %d, parent: %d, numchildren: %d, children: ",
		 i, SpanArray[i].parent, SpanArray[i].noofchildren); 
	for (j = 0; j < SpanArray[i].noofchildren; j++)
	     CmiPrintf("%d ",SpanArray[i].children[j]);
	CmiPrintf("\n");
    }
}



CmiSendToSpanTreeLeaves(size, msg)
int size;
char * msg;
{
    int node;

    /* node 0 cannot be a leaf of a spanning tree: it is the root */
    for (node = 1; node < numnodes; node++)
    if (SpanArray[node].noofchildren == 0)  /* it is a leaf */
        CmiAsyncSendFn(node, size, msg);
}
