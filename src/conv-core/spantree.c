#include <converse.h>

#if CMK_SPANTREE_USE_COMMON_CODE
/*************************************
 * This is used if the node level 
 * spanning tree is not available
 **************************************/
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


#if CMK_NODELEVEL_SPANTREE_AVAILABLE

/******************************
 * Node level spanning tree
 *****************************/

void 
CmiNodeSpanTreeInit(void) 
{
  return;
}


int
CmiNodeSpanTreeParent(int node)
{
  return (node ? ((node-1)/CMK_SPANTREE_MAXSPAN) : (-1));
}


int
CmiNodeSpanTreeRoot(void) 
{ 
  return 0; 
}


void 
CmiNodeSpanTreeChildren(int node, int *children)
{
  int i;

  for (i=1; i<=CMK_SPANTREE_MAXSPAN; i++)
    if (CMK_SPANTREE_MAXSPAN*node+i < CmiNumNodes())
      children[i-1] = node*CMK_SPANTREE_MAXSPAN+i;
    else 
      children[i-1] = (-1);
}


int
CmiNumNodeSpanTreeChildren(int node)
{
    if ((node+1)*CMK_SPANTREE_MAXSPAN < CmiNumNodes()) 
      return CMK_SPANTREE_MAXSPAN;
    else if (node*CMK_SPANTREE_MAXSPAN+1 >= CmiNumNodes()) 
           return 0;
         else 
           return ((CmiNumNodes()-1)-node*CMK_SPANTREE_MAXSPAN);
}


void
CmiSendToNodeSpanTreeLeaves(int size, char *msg)
{
  int node;

  for (node=(CmiNumNodes()-2)/CMK_SPANTREE_MAXSPAN;node<CmiNumNodes();node++)
    CmiSyncSend(node,size,msg);
}


void
PrintNodeSpanTree(void)
{
  int i,j;
  int children[CMK_SPANTREE_MAXSPAN];
  for (i=0; i<CmiNumNodes(); i++) {
    CmiPrintf("node: %d, parent: %d, numchildren: %d, children: ",
      i, CmiNodeSpanTreeParent(i), CmiNumSpanTreeChildren(i));
    CmiNodeSpanTreeChildren(i, children);
    for (j=0; j<CmiNumNodeSpanTreeChildren(i); j++)
      CmiPrintf("%d ", children[j]);
    CmiPrintf("\n");
  }
}

/***********************************************
 * Processor level spanning tree based on the 
 * node-level tree
 ***********************************************/
void 
CmiSpanTreeInit(void) 
{
  return;
}


int
CmiSpanTreeParent(int pe)
{
  int node=CmiNodeOf(pe);
  int peparent=((CmiNodeSpanTreeParent(node)==-1) ? -1 : CmiNodeFirst(CmiNodeSpanTreeParent(node))); 
  return(CmiRankOf(pe) ? CmiNodeFirst(node) : peparent);
}


int
CmiSpanTreeRoot(void) 
{ 
  return 0; 
}

/***********************************************
 * Notice that the size of the array children, varies
 * depending on the node size. The user should scan the
 * array untl he encounters a -1
 **********************************************/
void 
CmiSpanTreeChildren(int pe, int *children)
{
  int i, j;
  int node=CmiNodeOf(pe);
  for (j=1;j<CmiNodeSize(node);j++) {
    if (CmiRankOf(pe)) {
	children[i-1]=(-1);
	continue;
    }
    else {
	children[j-1]=pe+1;
    }
  }

  for (i=j-1; i<=CMK_SPANTREE_MAXSPAN; i++) {
    if (CmiRankOf(pe)) {
	children[i-1]=(-1);
	continue;
    }
    if (CMK_SPANTREE_MAXSPAN*node+i < CmiNumNodes())
      children[i-1] = node*CMK_SPANTREE_MAXSPAN+i;
    else 
      children[i-1] = (-1);
  }
}


int
CmiNumSpanTreeChildren(int pe)
{
    int node=CmiNodeOf(pe);
    if (CmiRankOf(pe)) return 0;

/* Code duplication - eliminates a function call */

    if ((node+1)*CMK_SPANTREE_MAXSPAN < CmiNumNodes()) 
      return(CmiNodeSize(node)-1+CMK_SPANTREE_MAXSPAN);
    else if (node*CMK_SPANTREE_MAXSPAN+1 >= CmiNumNodes()) 
           return(CmiNodeSize(node)-1+0);
         else 
           return ((CmiNumNodes()-1)-node*CMK_SPANTREE_MAXSPAN+CmiNodeSize(node)-1);
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
  int children[CMK_SPANTREE_MAXSPAN+4];
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

