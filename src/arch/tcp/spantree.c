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
	if (MAXSPAN * node + i < CmiNumPes())
	     children[i-1] = node * MAXSPAN + i;
	else children[i-1] = -1;
}


int
CmiNumSpanTreeChildren(node) int node; {
    if ((node + 1) * MAXSPAN < CmiNumPes()) return MAXSPAN;
    else if (node * MAXSPAN + 1 >= CmiNumPes()) return 0;
    else return ((CmiNumPes() - 1) - node * MAXSPAN);
}


CmiSendToSpanTreeLeaves(size, msg) int size; char * msg; {
    int node;

    for (node = (CmiNumPes() - 2) / MAXSPAN;   /* integer division */
	 node < CmiNumPes(); node++)
	CmiSyncSend(node,size,msg);
}



PrintSpanTree() {
    int i,j;
    int children[MAXSPAN];
    for (i = 0; i < CmiNumPes(); i++) {
	CmiPrintf("node: %d, parent: %d, numchildren: %d, children: ",
		 i, CmiSpanTreeParent(i), CmiNumSpanTreeChildren(i));
	CmiSpanTreeChildren(i, children);
	for (j = 0; j < CmiNumSpanTreeChildren(i); j++)
	     CmiPrintf("%d ", children[j]);
	CmiPrintf("\n");
    }
}
