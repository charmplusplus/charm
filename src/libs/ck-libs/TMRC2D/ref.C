#include "ref.h"
#include "refine.decl.h"
#include "messages.h"
#include "node.h"

extern CProxy_chunk mesh;

// nodeRef methods
node nodeRef::get()
{
  node n;
  nodeMsg *nm;
  intMsg *im = new intMsg;
  im->anInt = idx;
  nm = mesh[cid].getNode(im);
  n.set(nm->x, nm->y);
  CkFreeMsg(nm);
  return n;
}

void nodeRef::setBorder()
{
  intMsg *im = new intMsg;
  im->anInt = idx;
  mesh[cid].setBorder(im);
}

void nodeRef::update(node& m)
{
  nodeMsg *nm = new nodeMsg;
  nm->idx = idx;
  nm->x = m.X();
  nm->y = m.Y();
  mesh[cid].updateNodeCoords(nm);
}

void nodeRef::reportPos(node& m)
{
  nodeMsg *nm = new nodeMsg;
  nm->idx = idx;
  nm->x = m.X();
  nm->y = m.Y();
  mesh[cid].reportPos(nm);
}

void nodeRef::remove()
{
  intMsg *im = new intMsg;
  im->anInt = idx;
  mesh[cid].removeNode(im);
}

int nodeRef::lock()
{
  intMsg *im1 = new intMsg, *im2;
  int result;
  im1->anInt = idx;
  im2 = mesh[cid].lockNode(im1);
  result = im2->anInt;
  CkFreeMsg(im2);
  return result;
}

void nodeRef::unlock()
{
  intMsg *im = new intMsg;
  im->anInt = idx;
  mesh[cid].unlockNode(im);
}

int nodeRef::safeToMove(node& m, elemRef& E0, edgeRef& e0, edgeRef& e1, 
			nodeRef& n1, nodeRef& n2, nodeRef& n3)
{
  nodeMsg *nm = new nodeMsg;
  intMsg *im;
  int result;
  nm->idx = idx;
  nm->x = m.X(); nm->y = m.Y();
  im = mesh[cid].safeToMoveNode(nm);
  result = im->anInt;
  CkFreeMsg(im);
  return result;
}

// edgeRef methods
void edgeRef::update(nodeRef& oldval, nodeRef& newval)
{
  updateMsg *um = new updateMsg;
  um->idx = idx;
  um->oldval = oldval;
  um->newval = newval;
  mesh[cid].updateNode(um);
}

void edgeRef::update(elemRef& oldval, elemRef& newval)
{
  updateMsg *um = new updateMsg;
  um->idx = idx;
  um->oldval = oldval;
  um->newval = newval;
  mesh[cid].updateElement(um);
}

elemRef edgeRef::get(elemRef& m)
{
  refMsg *rm1 = new refMsg, *rm2;
  rm1->aRef = m;
  rm1->idx = idx;
  rm2 = mesh[cid].getNeighbor(rm1);
  elemRef result((elemRef&)rm2->aRef);
  CkFreeMsg(rm2);
  return result;
}

nodeRef edgeRef::get(nodeRef nr)
{
  refMsg *rm1 = new refMsg, *rm2;
  rm1->aRef = nr;
  rm1->idx = idx;
  rm2 = mesh[cid].getNotNode(rm1);
  nodeRef result((nodeRef&)rm2->aRef);
  CkFreeMsg(rm2);
  return result;
}

elemRef edgeRef::getNot(elemRef er)
{
  refMsg *rm1 = new refMsg, *rm2;
  rm1->aRef = er;
  rm1->idx = idx;
  rm2 = mesh[cid].getNotElem(rm1);
  elemRef result((elemRef&)rm2->aRef);
  CkFreeMsg(rm2);
  return result;
}

int edgeRef::setPending()
{
  intMsg *im1 = new intMsg, *im2;
  int result;
  im1->anInt = idx;
  im2 = mesh[cid].setPending(im1);
  result = im2->anInt;
  CkFreeMsg(im2);
  return result;
}

void edgeRef::unsetPending()
{
  intMsg *im = new intMsg;
  im->anInt = idx;
  mesh[cid].unsetPending(im);
}

int edgeRef::isPending()
{
  intMsg *im1 = new intMsg, *im2;
  int result;
  im1->anInt = idx;
  im2 = mesh[cid].isPending(im1);
  result = im2->anInt;
  CkFreeMsg(im2);
  return result;
}

void edgeRef::midpoint(node& result)
{
  nodeMsg *nm;
  intMsg *im = new intMsg;
  im->anInt = idx;
  nm = mesh[cid].midpoint(im);
  result.set(nm->x, nm->y);
  CkFreeMsg(nm);
}

void edgeRef::remove()
{
  intMsg *im = new intMsg;
  im->anInt = idx;
  mesh[cid].removeEdge(im);
}

int edgeRef::split(nodeRef *m, edgeRef *e_prime, nodeRef othernode, 
		   elemRef eRef)
{
  splitOutMsg *som;
  int result;
  splitInMsg *sim = new splitInMsg;
  sim->idx = idx;
  sim->n = othernode;
  sim->e = eRef;
  som = mesh[cid].split(sim);
  *m = som->n;
  *e_prime = som->e;
  result = som->result;
  CkFreeMsg(som);
  return result;
}

void edgeRef::checkPending(elemRef e)
{
  refMsg *rm = new refMsg;
  rm->idx = idx;
  rm->aRef = e;
  mesh[cid].checkPending(rm);
}

void edgeRef::checkPending(elemRef e, elemRef ne)
{
  drefMsg *rm = new drefMsg;
  rm->idx = idx;
  rm->aRef1 = e;
  rm->aRef2 = ne;
  mesh[cid].checkPending(rm);
}

// elemRef methods
double elemRef::getArea()
{
  doubleMsg *dm;
  double result;
  intMsg *im = new intMsg;
  im->anInt = idx;
  dm = mesh[cid].getArea(im);
  result = dm->aDouble;
  CkFreeMsg(dm);
  return result;
}

void elemRef::update(edgeRef& oldval, edgeRef& newval)
{
  updateMsg *um = new updateMsg;
  um->idx = idx;
  um->oldval = oldval;
  um->newval = newval;
  mesh[cid].updateElementEdge(um);
}

int elemRef::isLongestEdge(edgeRef& e)
{
  intMsg *im;
  int result;
  refMsg *rm = new refMsg;
  rm->aRef = e;
  rm->idx = idx;
  im = mesh[cid].isLongestEdge(rm);
  result = im->anInt;
  CkFreeMsg(im);
  return result;
}

void elemRef::update(edgeRef& e0, edgeRef& e1, edgeRef& e2)
{
  edgeUpdateMsg *em = new edgeUpdateMsg;
  em->idx = idx;
  em->e0 = e0;
  em->e1 = e1;
  em->e2 = e2;
  mesh[cid].updateEdges(em);
}

edgeRef elemRef::getEdge(edgeRef eR, nodeRef nR)
{
  collapseMsg *cm = new collapseMsg;
  refMsg *rm;
  cm->er = eR;
  cm->nr1 = nR;
  cm->idx = idx;
  rm = mesh[cid].getEdge(cm);
  edgeRef eRef(rm->aRef.cid, rm->aRef.idx);
  CkFreeMsg(rm);
  return eRef;
}

void elemRef::setTargetArea(double ta)
{
  doubleMsg *dm = new doubleMsg;
  dm->idx = idx;
  dm->aDouble = ta;
  mesh[cid].setTargetArea(dm);
}

void elemRef::resetTargetArea(double ta)
{
  doubleMsg *dm = new doubleMsg;
  dm->idx = idx;
  dm->aDouble = ta;
  mesh[cid].resetTargetArea(dm);
}

void elemRef::remove()
{
  intMsg *im = new intMsg;
  im->anInt = idx;
  mesh[cid].removeElement(im);
}

void elemRef::collapseHelp(edgeRef er, nodeRef nr1, nodeRef nr2)
{
  collapseMsg *cm = new collapseMsg;
  cm->er = er;
  cm->nr1 = nr1;
  cm->nr2 = nr2;
  cm->idx = idx;
  mesh[cid].collapseHelp(cm);
}
