#include <converse.h>

#define DEGREE 4

int CmiSpanTreeParent(node)
int node;
{
  return((node-1)/DEGREE);
}

int CmiSpanTreeRoot()
{
    return 0;  /* node 0 is always the root of the spanning tree */
}

int CmiNumSpanTreeChildren(node)
int node;
{
  int n=Cmi_numpes-1-node*DEGREE;
  if (n > DEGREE) {
	return(DEGREE);
  }

  if (n<0) return(0);

  return(n);
}

void CmiSpanTreeChildren(node, children)
int node, *children;
{
  CmiPrintf("NOT IMPLEMENTED\n");
}

void CmiSendToSpanTreeLeaves(size, msg)
int size;
char * msg;
{
  CmiPrintf("NOT IMPLEMENTED\n");
}
