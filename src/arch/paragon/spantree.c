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
