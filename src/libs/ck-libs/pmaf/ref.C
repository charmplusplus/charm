// Reference class for PMAF3D Framework
// Created by: Terry L. Wilmarth
#include "ref.h"
#include "node.h"
#include "PMAF.decl.h"
#include "messages.h"

extern CProxy_chunk mesh;

// nodeRef methods
node nodeRef::get()
{
  node n;
  nodeMsg *nm;
  nm = mesh[cid].getNode(idx);
  n.set(nm->coord[0], nm->coord[1], nm->coord[2]);
  CkFreeMsg(nm);
  return n;
}

void nodeRef::update(node& m)
{
  nodeMsg *nm = new nodeMsg;
  nm->idx = idx;
  nm->coord[0] = m.getCoord(0);
  nm->coord[1] = m.getCoord(1);
  nm->coord[2] = m.getCoord(2);
  mesh[cid].updateNodeCoord(nm);
}

// elemRef methods
double elemRef::getVolume()
{
  doubleMsg *dm;
  double result;
  if (idx > 100000) 
    CkPrintf("---------------> idx is %d before getVolume!\n", idx);
  intMsg *im = new intMsg;
  im->anInt = idx;
  dm = mesh[cid].getVolume(im);
  result = dm->aDouble;
  CkFreeMsg(dm);
  return result;
}

void elemRef::setTargetVolume(double ta)
{
  doubleMsg *dm = new doubleMsg;
  dm->idx = idx;
  dm->aDouble = ta;
  mesh[cid].setTargetVolume(dm);
}

void elemRef::resetTargetVolume(double ta)
{
  doubleMsg *dm = new doubleMsg;
  dm->idx = idx;
  dm->aDouble = ta;
  mesh[cid].resetTargetVolume(dm);
}
