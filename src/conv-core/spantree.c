#include <converse.h>

#if CMK_SPANTREE_USE_COMMON_CODE

void 
CmiSpanTreeInit(void) 
{
  return;
}


int
CmiSpanTreeParent(int node)
{
  return (node ? ((node-1)/CMK_SPANTREE_MAXSPAN) : (-1));
}


int
CmiSpanTreeRoot(void) 
{ 
  return 0; 
}


void 
CmiSpanTreeChildren(int node, int *children)
{
  int i;

  for (i=1; i<=CMK_SPANTREE_MAXSPAN; i++)
    if (CMK_SPANTREE_MAXSPAN*node+i < CmiNumPes())
      children[i-1] = node*CMK_SPANTREE_MAXSPAN+i;
    else 
      children[i-1] = (-1);
}


int
CmiNumSpanTreeChildren(int node)
{
    if ((node+1)*CMK_SPANTREE_MAXSPAN < CmiNumPes()) 
      return CMK_SPANTREE_MAXSPAN;
    else if (node*CMK_SPANTREE_MAXSPAN+1 >= CmiNumPes()) 
           return 0;
         else 
           return ((CmiNumPes()-1)-node*CMK_SPANTREE_MAXSPAN);
}


void
CmiSendToSpanTreeLeaves(int size, char *msg)
{
  int node;

  for (node=(CmiNumPes()-2)/CMK_SPANTREE_MAXSPAN;node<CmiNumPes();node++)
    CmiSyncSend(node,size,msg);
}


void
PrintSpanTree(void)
{
  int i,j;
  int children[CMK_SPANTREE_MAXSPAN];
  for (i=0; i<CmiNumPes(); i++) {
    CmiPrintf("node: %d, parent: %d, numchildren: %d, children: ",
      i, CmiSpanTreeParent(i), CmiNumSpanTreeChildren(i));
    CmiSpanTreeChildren(i, children);
    for (j=0; j<CmiNumSpanTreeChildren(i); j++)
      CmiPrintf("%d ", children[j]);
    CmiPrintf("\n");
  }
}

#endif
