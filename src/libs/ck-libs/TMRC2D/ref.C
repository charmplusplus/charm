#include "ref.h"
#include "refine.decl.h"
#include "node.h"
#include "messages.h"

extern CProxy_chunk mesh;

// edgeRef methods
void edgeRef::update(elemRef& oldval, elemRef& newval)
{
  mesh[cid].updateElement(idx, oldval, newval);
}

elemRef edgeRef::get(elemRef& m)
{
  refMsg *rm2;
  rm2 = mesh[cid].getNeighbor(idx, m);
  elemRef result((elemRef&)rm2->aRef);
  CkFreeMsg(rm2);
  return result;
}

elemRef edgeRef::getNot(elemRef er)
{
  refMsg *rm2;
  rm2 = mesh[cid].getNotElem(idx, er);
  elemRef result((elemRef&)rm2->aRef);
  CkFreeMsg(rm2);
  return result;
}

void edgeRef::remove()
{
  mesh[cid].removeEdge(idx);
}

int edgeRef::split(int *m, edgeRef *e_prime, node iNode, node fNode,
		   elemRef requester, int *local, int *first, int *nullNbr)
{
  splitOutMsg *som;
  int result;
  som = mesh[cid].split(idx, requester, iNode, fNode);
  *m = som->n;
  *e_prime = som->e;
  *local = som->local;
  *first = som->first;
  result = som->result;
  CkFreeMsg(som);
  return result;
}

void edgeRef::setPending()
{
  mesh[cid].setPending(idx);
}

void edgeRef::checkPending(elemRef e)
{
  mesh[cid].checkPending(idx, e);
}

void edgeRef::checkPending(elemRef e, elemRef ne)
{
  mesh[cid].checkPending(idx, e, ne);
}

// elemRef methods
double elemRef::getArea()
{
  doubleMsg *dm;
  double result;
  dm = mesh[cid].getArea(idx);
  result = dm->aDouble;
  CkFreeMsg(dm);
  return result;
}

void elemRef::update(edgeRef& oldval, edgeRef& newval)
{
  mesh[cid].updateElementEdge(idx, oldval, newval);
}

int elemRef::isLongestEdge(edgeRef& e)
{
  intMsg *im;
  int result;
  im = mesh[cid].isLongestEdge(idx, e);
  result = im->anInt;
  CkFreeMsg(im);
  return result;
}

void elemRef::update(edgeRef& e0, edgeRef& e1, edgeRef& e2)
{
  mesh[cid].updateEdges(idx, e0, e1, e2);
}

void elemRef::setTargetArea(double ta)
{
  mesh[cid].setTargetArea(idx, ta);
}

void elemRef::resetTargetArea(double ta)
{
  mesh[cid].resetTargetArea(idx, ta);
}

void elemRef::remove()
{
  mesh[cid].removeElement(idx);
}

//void elemRef::collapseHelp(edgeRef er, node n1, node n2)
//{
//  mesh[cid].collapseHelp(idx, er, n1, n2);
//}
