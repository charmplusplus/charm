// Element class for PMAF3D Framework
// Created by: Terry L. Wilmarth
#include "element.h"
#include "chunk.h"
#include "matrix.h"

void element::setTargetVolume(double volume) { 
  if ((volume < targetVolume) || (targetVolume < 0.0)) targetVolume = volume;
}

void element::update(nodeRef& oldval, nodeRef& newval)
{
  if (nodes[0] == oldval) nodes[0] = newval;
  else if (nodes[1] == oldval) nodes[1] = newval;
  else if (nodes[2] == oldval) nodes[2] = newval;
  else if (nodes[3] == oldval) nodes[3] = newval;
  else CkPrintf("ERROR: element::update: oldval not found\n");
}

int element::hasNode(nodeRef n)
{
  return ((n==nodes[0]) || (n==nodes[1]) || (n==nodes[2]) || (n==nodes[3]));
}

int element::hasNode(node n)
{
  node nodeCoords[4];
  for (int i=0; i<4; i++) nodeCoords[i] = C->theNodes[nodes[i].idx];
  return ((n==nodeCoords[0]) || (n==nodeCoords[1]) || (n==nodeCoords[2]) 
	  || (n==nodeCoords[3]));
}

int element::hasNodes(nodeRef n1, nodeRef n2, nodeRef n3)
{
  return (hasNode(n1) && hasNode(n2) && hasNode(n3));
}

int element::hasNodes(double nodeCoords[3][3])
{
  node inNodes[3];
  inNodes[0].set(nodeCoords[0][0], nodeCoords[0][1], nodeCoords[0][2]);
  inNodes[1].set(nodeCoords[1][0], nodeCoords[1][1], nodeCoords[1][2]);
  inNodes[2].set(nodeCoords[2][0], nodeCoords[2][1], nodeCoords[2][2]);
  return (hasNode(inNodes[0]) && hasNode(inNodes[1]) && hasNode(inNodes[2]));
}

int element::getNode(node n) {
  for (int i=0; i<4; i++)
    if (C->theNodes[nodes[i].idx] == n)
      return i;
  return -1;
}

int element::getNodeIdx(nodeRef n)
{
  if (nodes[0] == n) return 0;
  else if (nodes[1] == n) return 1;
  else if (nodes[2] == n) return 2;
  else if (nodes[3] == n) return 3;
  else { 
    CkPrintf("ERROR: element::getNodeIdx: nodeRef n not found on element.\n");
    return -1;
  }
}

int element::checkFace(node n1, node n2, node n3, elemRef nbr)
{
  int a=-1, b=-1, c=-1, d=-1;
  elemRef abc, abd, acd, bcd;
  double Aabc, Aabd, Aacd, Abcd;
  // align nodes with requester
  a = getNode(n1);
  b = getNode(n2);
  c = getNode(n3);
  d = 6-a-b-c;
  abc = faceElements[a+b+c-3];
  abd = faceElements[a+b+d-3];
  acd = faceElements[a+c+d-3];
  bcd = faceElements[b+c+d-3];
  CmiAssert(abc == nbr);
  Aabc = getArea(a, b, c);
  Aabd = getArea(a, b, d);
  Aacd = getArea(a, c, d);
  Abcd = getArea(b, c, d);
  if ((Aabc >= Aabd) && (Aabc >= Aacd) && (Aabc >= Abcd))
    return 1;
  mesh[myRef.cid].refineElement(myRef.idx, getVolume()/2.0);
  return 0;
}

double element::getVolume()
{ // get a cached volume calculation; if none, computes it
  if (currentVolume < 0.0)
    calculateVolume();
  return currentVolume;
}

void element::calculateVolume()
{ // calulate volume of tetrahedron
  Matrix mtx(4);
  int i, j;
  
  // Initialize the matrix with this element's coordinates
  for (i=0; i<4; i++)  // for each node
    for (j=0; j<3; j++)  // for each coord of that node
      mtx.setElement(i, j, C->theNodes[nodes[i].idx].getCoord(j));
  for (i=0; i<4; i++)  // for each node
    mtx.setElement(i, 3, 1.0); // set end of row to 1
  currentVolume = mtx.determinant() / 6.0;
  if (currentVolume < 0.0) currentVolume *= -1.0;
}

double element::getArea(int n1, int n2, int n3)
{ // calulate area of triangle using Heron's formula:
  // Let a, b, c be the lengths of the three sides.
  // Area=SQRT(s(s-a)(s-b)(s-c)), where s=(a+b+c)/2 or perimeter/2.
  node n[3];
  double s, perimeter, len[3];
  n[0] = C->theNodes[nodes[n1].idx];
  n[1] = C->theNodes[nodes[n2].idx];
  n[2] = C->theNodes[nodes[n3].idx];
  // fine lengths of sides
  len[0] = n[0].distance(n[1]);
  len[1] = n[0].distance(n[2]);
  len[2] = n[1].distance(n[2]);
  // apply Heron's formula
  perimeter = len[0] + len[1] + len[2];
  s = perimeter / 2.0;
  // cache the result in currentArea
  return(sqrt(s * (s - len[0]) * (s - len[1]) * (s - len[2])));
}

double element::findLongestEdge(int *le1, int *le2, int *nl1, int *nl2)
{
  double maxLength = 0.0, length;
  for (int i=0; i<4; i++)
    for (int j=i+1; j<4; j++) {
      length = C->theNodes[nodes[i].idx].distance(C->theNodes[nodes[j].idx]);
      if (length > maxLength) {
	maxLength = length;
	*le1 = i; *le2 = j;
      }
    }
  if (*le1 == 0) {
    if (*le2 == 1) { *nl1 = 2; *nl2 = 3; }
    else if (*le2 == 2) { *nl1 = 1; *nl2 = 3; }
    else { *nl1 = 1; *nl2 = 2; }
  }
  else if (*le1 == 1) {
    if (*le2 == 2) { *nl1 = 0; *nl2 = 3; }
    else { *nl1 = 0; *nl2 = 2; }
  }
  else { *nl1 = 0; *nl2 = 1; }
  return maxLength;
}

// Delaunay operations
// perform 2->3 flip on this element with element neighboring on face
void element::flip23(int face[3])
{
  // call face abc, and the other node on this element d, other node on
  // neighboring element e
  int a = face[0], b = face[1], c = face[2], d = 6 - (face[0]+face[1]+face[2]);
  int ap, bp, cp, ep;  // node indices on neighbor
  // get neighbor ref and adjacent element refs
  elemRef nbrRef = getFace(a, b, c), acdNbr = getFace(a, c, d), 
    bcdNbr = getFace(b, c, d);
  elemRef abeNbr, aceNbr, bceNbr, newElem;

  if (nbrRef.cid == myRef.cid) { // neighbor is local
    // get node indices relative to neighbor
    ap = C->theElements[nbrRef.idx].getNodeIdx(nodes[a]);
    bp = C->theElements[nbrRef.idx].getNodeIdx(nodes[b]);
    cp = C->theElements[nbrRef.idx].getNodeIdx(nodes[c]);
    ep = 6 - (ap+bp+cp);
    // get elemRefs for neighbors of neighbor element
    abeNbr = C->theElements[nbrRef.idx].getFace(ap, bp, ep);
    aceNbr = C->theElements[nbrRef.idx].getFace(ap, cp, ep);
    bceNbr = C->theElements[nbrRef.idx].getFace(bp, cp, ep);
    // add new element as (a,c,d,e)
    newElem = C->addElement(nodes[a], nodes[c], nodes[d], 
			    C->theElements[nbrRef.idx].nodes[ep]);
    // initialize the new elements neighbors on faces
    C->theElements[newElem.idx].faceElements[0] = acdNbr;
    C->theElements[newElem.idx].faceElements[1] = aceNbr;
    C->theElements[newElem.idx].faceElements[2] = myRef;
    C->theElements[newElem.idx].faceElements[3] = nbrRef;
    // this element becomes (a,b,d,e)
    nodes[c] = C->theElements[nbrRef.idx].nodes[ep];
    // fix some of the local face neighbors
    faceElements[0] = abeNbr;
    faceElements[2] = newElem;
    faceElements[3] = nbrRef;
    // the neighbor element becomes (b,c,d,e)
    C->theElements[nbrRef.idx].nodes[ap] = nodes[d];
    // fix some of the neighbor's face neighbors
    C->theElements[nbrRef.idx].setFace(ap, bp, cp, bcdNbr);
    C->theElements[nbrRef.idx].setFace(ap, bp, ep, myRef);
    C->theElements[nbrRef.idx].setFace(ap, cp, ep, newElem);
    // make adjacent elements refer back to the new element arrangement
    mesh[acdNbr.cid].updateFace(acdNbr.idx, myRef, newElem);
    mesh[aceNbr.cid].updateFace(aceNbr.idx, nbrRef, newElem);
    mesh[abeNbr.cid].updateFace(abeNbr.idx, nbrRef, myRef);
    mesh[bcdNbr.cid].updateFace(bcdNbr.idx, myRef, nbrRef);
  }
  else { // neighbor is on remote chunk
    flip23request *fr = new flip23request;
    flip23response *rf;

    // pack up configuration information for neighbor
    fr->a = C->theNodes[nodes[a].idx];
    fr->b = C->theNodes[nodes[b].idx];
    fr->c = C->theNodes[nodes[c].idx];
    fr->d = C->theNodes[nodes[d].idx];
    fr->acd = acdNbr;
    fr->bcd = bcdNbr;
    fr->requester = myRef;
    fr->requestee = nbrRef.idx;
    // neighbor will create (a,c,d,e) and become (b,c,d,e)
    rf = mesh[nbrRef.cid].flip23remote(fr);
    
    // acd and bcd faces have moved to the other chunk, so
    // if they are surface faces here, remove them
    if (getFace(a, c, d).idx == -1)
      C->removeFace(nodes[a].idx, nodes[c].idx, nodes[d].idx);
    if (getFace(b, c, d).idx == -1)
      C->removeFace(nodes[b].idx, nodes[c].idx, nodes[d].idx);

    // does e, the neighbor's other node, have an incarnation here?
    nodeRef eRef = C->findNode(rf->e);
    if (eRef.idx == -1)  // NO!
      eRef = C->addNode(rf->e);  // add e to this chunk

    // this element becomes (a,b,d,e)
    nodes[c] = eRef;
    // fix the local face neighbors
    setFace(a, b, c, rf->abe);
    setFace(a, c, d, rf->acde);
    setFace(b, c, d, nbrRef);
    if (rf->abe.idx == -1)  // if new face (a,b,e) is on surface
      C->addFace(nodes[a].idx, nodes[b].idx, nodes[c].idx);  // add to surface
    // make adjacent elements refer back to new element arrangement
    mesh[acdNbr.cid].updateFace(acdNbr.idx, myRef, rf->acde);
    mesh[bcdNbr.cid].updateFace(bcdNbr.idx, myRef, nbrRef);
  }
}

flip23response *element::flip23remote(flip23request *fr)
{
  int a, b, c, e;
  elemRef newElem, abeNbr, aceNbr;
  flip23response *rf = new flip23response;

  // does d, the neighbor's other node, have an incarnation here?
  nodeRef dRef = C->findNode(fr->d);
  if (dRef.idx == -1)  // NO!
    dRef = C->addNode(fr->d);  // add d
  
  // match local a,b,c indices to those on neighbor element; derive e
  for (int i=0; i<4; i++)
    if (C->theNodes[nodes[i].idx] == fr->a)
      a = i;
    else if (C->theNodes[nodes[i].idx] == fr->b)
      b = i;
    else if (C->theNodes[nodes[i].idx] == fr->c)
      c = i;
  e = 6 - (a + b + c);
  abeNbr = getFace(a, b, e);
  aceNbr = getFace(a, c, e);

  // (a,b,e) face will move to other chunk; if on surface here, remove it
  if (getFace(a, b, e).idx == -1)
    C->removeFace(nodes[a].idx, nodes[b].idx, nodes[e].idx);
  // faces (a,c,d) and (b,c,d) are new to this chunk; if they are surface
  // faces, add to surface here
  if (fr->acd.idx == -1)
    C->addFace(nodes[a].idx, nodes[c].idx, dRef.idx);
  if (fr->bcd.idx == -1)
    C->addFace(nodes[b].idx, nodes[c].idx, dRef.idx);

  // add new element for (a,c,d,e)
  newElem = C->addElement(nodes[a], nodes[c], dRef, nodes[e]);
  // initialize new element's face neighbors
  C->theElements[newElem.idx].setFace(0, fr->acd);
  C->theElements[newElem.idx].setFace(1, aceNbr);
  C->theElements[newElem.idx].setFace(2, fr->requester);
  C->theElements[newElem.idx].setFace(3, myRef);

  // this element becomes (b,c,d,e)
  nodes[a] = dRef;
  // fix local face neighbors
  setFace(a, b, c, fr->bcd);
  setFace(a, b, e, fr->requester);
  setFace(a, c, e, newElem);

  // make adjacent elements refer back to new element arrangement
  mesh[abeNbr.cid].updateFace(abeNbr.idx, myRef, fr->requester);
  mesh[aceNbr.cid].updateFace(aceNbr.idx, myRef, newElem);

  // prepare response for neighbor
  rf->acde = newElem;
  rf->abe = abeNbr;
  rf->e = C->theNodes[nodes[e].idx];

  CkFreeMsg(fr);
  return rf;
}

// perform 3->2 flip on this element with elements neighboring on edge
void element::flip32(int edge[2]) 
{ // let this element be abde, where de is the input edge
  int a = 0, b, d = edge[0], e = edge[1];
  elemRef acde, bcde, abeNbr;

  // get node indices
  if ((d == 0) || (e == 0)) {
    a = 1;
    if ((d == 1) || (e == 1)) {
      a = 2;
      if ((d == 2) || (e == 2))  a = 3;
    }
  }
  b = 6 - (a + d + e);

  // get neighbors involved in flip
  acde = getFace(a, d, e);
  bcde = getFace(b, d, e);
  // get face neighbors
  abeNbr = getFace(a, b, e);

  // bcde will cease to exist; get face neighbor info for bcd and bce
  // and remove bcde
  flip32request *fr1 = new flip32request;
  flip32response *rf1;
  fr1->requestee = bcde.idx;
  fr1->requester = myRef;
  fr1->b = C->theNodes[nodes[b].idx];
  fr1->d = C->theNodes[nodes[d].idx];
  fr1->e = C->theNodes[nodes[e].idx];
  if (bcde.cid == myRef.cid)  // bcde is local
    rf1 = C->theElements[bcde.idx].remove32element(fr1);
  else rf1 = mesh[bcde.cid].remove32element(fr1);  // bcde is remote
  // rf1 now contains bcd and bce face neighbors and node c

  // acde will change to abce; get face neighbor info for acd and tell 
  // it about abe and bce
  flip32request *fr2 = new flip32request;
  flip32response *rf2;
  fr2->requestee = acde.idx;
  fr2->bcde = bcde;
  fr2->requester = myRef;
  fr2->a = C->theNodes[nodes[a].idx];
  fr2->d = C->theNodes[nodes[d].idx];
  fr2->e = C->theNodes[nodes[e].idx];
  fr2->abe = abeNbr;
  fr2->bce = rf1->bce;
  if (acde.cid == myRef.cid)  // acde is local
    rf2 = C->theElements[acde.idx].flip32remote(fr2);
  else rf2 = mesh[acde.cid].flip32remote(fr2);  // acde is remote
  // rf2 now contains acd face neighbor and node c
  
  // check if neighbors agree about node c
  if ( !(rf1->c == rf2->c) )  
    CkPrintf("ERROR: flip32: Neighbors disagree about node c.\n");
  
  // does c have an incarnation here?
  nodeRef cRef = C->findNode(rf1->c);
  if (cRef.idx == -1)  // NO!
    cRef = C->addNode(rf1->c);   // add c
  
  // if abe was on surface and is now remote, remove from surface
  if ((abeNbr.idx == -1) && (acde.cid != myRef.cid))
    C->removeFace(nodes[a].idx, nodes[b].idx, nodes[e].idx);
  // if acd or bcd were remote and are on the surface, add to surface here
  if ((rf2->acd.idx == -1) && (acde.cid != myRef.cid))
    C->addFace(nodes[a].idx, cRef.idx, nodes[d].idx);
  if ((rf1->bcd.idx == -1) && (bcde.cid != myRef.cid))
    C->addFace(nodes[b].idx, cRef.idx, nodes[d].idx);
  
  // this element becomes (a,b,c,d)
  nodes[e] = cRef;
  // fix the local face neighbors
  setFace(a, b, e, acde);
  setFace(a, d, e, rf2->acd);
  setFace(b, d, e, rf1->bcd);
  // make adjacent elements refer back to new element arrangement
  mesh[rf2->acd.cid].updateFace(rf2->acd.idx, acde, myRef);
  mesh[rf1->bcd.cid].updateFace(rf1->bcd.idx, bcde, myRef);
}

flip32response *element::flip32remote(flip32request *fr)
{
  // this element is acde
  flip32response *rf = new flip32response();
  int a, b, c, d, e;
  elemRef acdNbr;
  
  // extract a, d and e indices and acdNbr
  for (int i=0; i<4; i++) {
    if (C->theNodes[nodes[i].idx] == fr->a)
      a = i;
    else if (C->theNodes[nodes[i].idx] == fr->d)
      d = i;
    else if (C->theNodes[nodes[i].idx] == fr->e)
      e = i;
    else 
      c = i;
  }
  acdNbr = getFace(a, c, d);

  // build the response: abde wants to know about acd face and node c
  rf->acd = acdNbr;
  rf->c = C->theNodes[nodes[c].idx];

  // does b, the node to switch d to, have an incarnation here?
  nodeRef bRef = C->findNode(fr->b);
  if (bRef.idx == -1)  // NO!
    bRef = C->addNode(fr->b);  // add b

  // this element becomes (a,b,c,e)
  b = d; 
  nodes[b] = bRef;
  setFace(a, b, e, fr->abe);
  setFace(b, c, e, fr->bce);
  setFace(a, b, c, fr->requester);

  // (a, c, d) face will move to other chunk; if on surface here, remove it
  if (acdNbr.idx == -1)
    C->removeFace(nodes[a].idx, nodes[c].idx, nodes[d].idx);
  // faces (a,b,e) and (b,c,e) are new to this chunk; if they are surface
  // faces, add to surface here
  if (fr->abe.idx == -1)
    C->addFace(nodes[a].idx, nodes[b].idx, nodes[e].idx);
  if (fr->bce.idx == -1)
    C->addFace(nodes[b].idx, nodes[c].idx, nodes[e].idx);

  // make adjacent elements refer back to new element arrangement
  mesh[fr->abe.cid].updateFace(fr->abe.idx, fr->requester, myRef);
  mesh[fr->bce.cid].updateFace(fr->bce.idx, fr->bcde, myRef);

  CkFreeMsg(fr);
  return rf;
}

flip32response *element::remove32element(flip32request *fr)
{
  // this element is bcde
  flip32response *rf = new flip32response;
  int b, c, d, e;
  elemRef bcdNbr, bceNbr;

  // get indices for b, c, d and e and neighbor elemRefs for bcd and bce
  for (int i=0; i<4; i++) {
    if (C->theNodes[nodes[i].idx] == fr->b)
      b = i;
    else if (C->theNodes[nodes[i].idx] == fr->d)
      d = i;
    else if (C->theNodes[nodes[i].idx] == fr->e)
      e = i;
    else 
      c = i;
  }
  bcdNbr = getFace(b, c, d);
  bceNbr = getFace(b, c, e);

  // build the response: abde wants to know about faces bcd, bce and node c
  rf->bcd = bcdNbr;
  rf->bce = bceNbr;
  rf->c = C->theNodes[nodes[c].idx];
  
  // (b,c,d) and (b,c,e) faces will move to other chunk; if they are
  // surface faces here, remove them
  if (bcdNbr.idx == -1)
    C->removeFace(nodes[b].idx, nodes[c].idx, nodes[d].idx);
  if (bceNbr.idx == -1)
    C->removeFace(nodes[b].idx, nodes[c].idx, nodes[e].idx);
  
  // remove this element
  present = 0;
  // NEED TO MAKE ELEMENT CREATIONS RECYCLE DELETED ELEMENTS

  CkFreeMsg(fr);
  return rf;
}

// test if this element should perform 2->3 flip with element neighboring 
// on face
int element::test23(int face[3])
{
  return 0;
}

// test if this element should perform 3->2 flip with elements neighboring 
// on edge
int element::test32(int edge[2])
{
  return 0;
}

int element::connectTest()
{
  intMsg *result;
  for (int i=0; i<4; i++) {
    // check if element of faceElement[i] has this element as a neighbor
    result = mesh[faceElements[i].cid].checkFace(faceElements[i].idx, myRef);
    if (!result->anInt) return 0;
    CkPrintf("Face %d is fine...\n", i);
  }
  return 1;
}
  
int element::hasFace(elemRef face)
{
  return((faceElements[0] == face) || (faceElements[1] == face) ||
	 (faceElements[2] == face) || (faceElements[3] == face));
}

// Largest face methods
void element::refineLF()
{
  static int start = 0; // start with a different face each time (if areas =)
  int lf, a, b, c, d;
  node n1, n2, n3, n4, centerpoint;
  elemRef abc, acd, bcd;
  double f[4], length; // to store face areas

  // Check if neighbor on largest face needs refinement
  // find largest face
  f[0] = getArea(0,1,2);
  f[1] = getArea(0,1,3);
  f[2] = getArea(0,2,3);
  f[3] = getArea(1,2,3);
  lf = start;
  for (int i=0; i<4; i++) if (f[i] > f[lf]) lf = i;
  // make abc largest face
  a = (lf == 3) ? 1 : 0; 
  b = (lf > 1) ? 2 : 1; 
  c = (lf == 0) ? 2 : 3;
  CmiAssert(a+b+c-3 == lf);
  abc = faceElements[a+b+c-3];
  // check abc on neighbor
  if (abc.cid != -1) {
    n1 = C->theNodes[nodes[a].idx];
    n2 = C->theNodes[nodes[b].idx];
    n3 = C->theNodes[nodes[c].idx];
    intMsg *im = mesh[abc.cid].checkFace(abc.idx, n1, n2, n3, myRef);
    if (!im->anInt) {
      CkFreeMsg(im);
      return;
    }
  }

  //CkPrintf("Refine: %d on %d: volume=%lf target=%lf\n",
  //   myRef.idx, myRef.cid, currentVolume, targetVolume);
  if ((currentVolume < targetVolume) || (currentVolume == 0.0) 
      || (targetVolume == 0.0)) { 
    return;
  }

  // lock this chunk first
  int iResult;
  intMsg *result;
  length = findLongestEdge(&a, &b, &c, &d);
  iResult = C->lockLocalChunk(myRef.cid, length);
  while (iResult != 1) {
    if (iResult == 0) return;
    else if (iResult == -1) {
      CthYield();
      length = findLongestEdge(&a, &b, &c, &d);
      iResult = C->lockLocalChunk(myRef.cid, length);
    }
  }
  // find largest face
  f[0] = getArea(0,1,2);
  f[1] = getArea(0,1,3);
  f[2] = getArea(0,2,3);
  f[3] = getArea(1,2,3);
  lf = start;
  for (int i=0; i<4; i++) if (f[i] > f[lf]) lf = i;
  start = (start+1) % 4;
  // make abc largest face
  a = (lf == 3) ? 1 : 0; 
  b = (lf > 1) ? 2 : 1; 
  c = (lf == 0) ? 2 : 3;
  d = 3 - lf;
  CmiAssert(a+b+c-3 == lf);
  abc = faceElements[a+b+c-3];
  acd = faceElements[a+c+d-3];
  bcd = faceElements[b+c+d-3];

  CkPrintf("-> Refine STATUS: a=%d b=%d c=%d d=%d abc=[%d,%d](fe%d) acd=[%d,%d](fe%d) bcd=[%d,%d](fe%d)\n", a, b, c, d, abc.idx, abc.cid, a+b+c-3, acd.idx, acd.cid, a+c+d-3, bcd.idx, bcd.cid, b+c+d-3);

  // build local locking cloud
  if (acd.cid != -1) {
    result = mesh[acd.cid].lockChunk(myRef.cid, length);
    while (result->anInt != 1) {
      if (result->anInt == 0) {
	C->unlockLocalChunk(myRef.cid);
	CkFreeMsg(result);
	return;
      }
      else if (result->anInt == -1) {
	CkFreeMsg(result);
	CthYield();
	result = mesh[acd.cid].lockChunk(myRef.cid, length);
      }
    }
    CkFreeMsg(result);
  }
  // acd is locked
  if (bcd.cid != -1) {
    result = mesh[bcd.cid].lockChunk(myRef.cid, length);
    while (result->anInt != 1) {
      if (result->anInt == 0) {
	C->unlockLocalChunk(myRef.cid);
	if (acd.cid != -1) mesh[acd.cid].unlockChunk(myRef.cid);
	CkFreeMsg(result);
	return;
      }
      else if (result->anInt == -1) {
	CkFreeMsg(result);
	CthYield();
	result = mesh[bcd.cid].lockChunk(myRef.cid, length);
      }
    }
    CkFreeMsg(result);
  }
  // bcd is locked
  n1 = C->theNodes[nodes[a].idx];
  n2 = C->theNodes[nodes[b].idx];
  n3 = C->theNodes[nodes[c].idx];
  n4 = C->theNodes[nodes[d].idx];
  if (abc.cid != -1) {
    CkPrintf("  -> Refine: %d on %d: Locking %d on %d.\n", myRef.idx, 
	     myRef.cid, abc.idx, abc.cid);
    result = mesh[abc.cid].lockLF(abc.idx, n1, n2, n3, n4, myRef, length);
    if (result->anInt == 0) {
      //CkPrintf("  -> Refine: %d on %d: Lock %d on %d FAILED.\n", myRef.idx, 
      //       myRef.cid, abc.idx, abc.cid);
      CkFreeMsg(result);
      C->unlockLocalChunk(myRef.cid);
      if (acd.cid != -1) mesh[acd.cid].unlockChunk(myRef.cid);
      if (bcd.cid != -1) mesh[bcd.cid].unlockChunk(myRef.cid);
      return;
    }
    //CkPrintf("  -> Refine: %d on %d: Lock %d on %d SUCCEEDED.\n", myRef.idx, 
    //     myRef.cid, abc.idx, abc.cid);
    CkFreeMsg(result);
  }
  // abc and neighbors locked
  // get nodeRefs for nodes
  //CkPrintf("  -> Refine: %d on %d: Cloud locked.\n", myRef.idx, myRef.cid);
  // test if face is on surface of mesh
  if (abc.cid == -1) { // face is on surface
    //CkPrintf("   -> Refine %d on %d (%lf -> %lf): face on surface\n", 
    //     myRef.idx, myRef.cid, getVolume(), getTargetVolume());
    // largest face is on surface of mesh
    // add new node in center of face and 1->3 split the tet
    centerpoint.set((n1.getCoord(0)+n2.getCoord(0)+n3.getCoord(0))/3.0,
		    (n1.getCoord(1)+n2.getCoord(1)+n3.getCoord(1))/3.0,
		    (n1.getCoord(2)+n2.getCoord(2)+n3.getCoord(2))/3.0);
    centerpoint.setSurface();    
    centerpoint.notFixed();
    nodeRef newNode = C->addNode(centerpoint);
    // modify this tet and add two new ones
    elemRef newElem1 = C->addElement(nodes[a],newNode,nodes[c],nodes[d]);
    elemRef newElem2 = C->addElement(newNode,nodes[b],nodes[c],nodes[d]);
    // need to updateFace on elements on acd and bcd faces: abd face
    // still on this element so no changes needed
    if (acd.cid != -1)
      mesh[acd.cid].updateFace(acd.idx, myRef, newElem1);
    if (bcd.cid != -1)
      mesh[bcd.cid].updateFace(bcd.idx, myRef, newElem2);
    // set faces for the new elements
    C->theElements[newElem1.idx].faceElements[0] = abc;
    C->theElements[newElem1.idx].faceElements[1] = myRef;
    C->theElements[newElem1.idx].faceElements[2] = acd;
    C->theElements[newElem1.idx].faceElements[3] = newElem2;
    C->theElements[newElem2.idx].faceElements[0] = abc;
    C->theElements[newElem2.idx].faceElements[1] = myRef;
    C->theElements[newElem2.idx].faceElements[2] = newElem1;
    C->theElements[newElem2.idx].faceElements[3] = bcd;
    // update my faces with the two new elements
    faceElements[a+c+d-3] = newElem1;
    faceElements[b+c+d-3] = newElem2;
    // pass target volume on to the two new elements
    C->theElements[newElem1.idx].setTargetVolume(targetVolume);
    C->theElements[newElem2.idx].setTargetVolume(targetVolume);
    // update surface information
    C->updateFace(nodes[a].idx, nodes[b].idx, nodes[c].idx,
		  nodes[c].idx, newNode.idx);
    C->addFace(nodes[b].idx, newNode.idx, nodes[c].idx);
    C->addFace(nodes[a].idx, newNode.idx, nodes[c].idx);
    // update local nodes with new node
    nodes[c] = newNode;
    calculateVolume(); // since this element has changed in size
    //CkPrintf("   <- Refine: Done w/ surface element refine of %d on %d\n",
    //     myRef.idx, myRef.cid);
  }
  else { // largest face is not on surface
    //CkPrintf("   -> Refine: %d on %d (%lf -> %lf): face NOT on surface; trying to split %d on %d\n", 
    //     myRef.idx, myRef.cid, getVolume(), getTargetVolume(), 
    //     abc.idx, abc.cid);
    splitResponse *theResult = 
      mesh[abc.cid].splitLF(abc.idx, n1, n2, n3, n4, myRef);
    //CkPrintf("    -> Split: result=%d\n", theResult->success);
    if (theResult->success == 1) { // successfully split neighbor
      // extract newNode
      node newNode(theResult->newNode[0], theResult->newNode[1], 
		   theResult->newNode[2]);
      newNode.notFixed();
      newNode.notSurface();
      nodeRef newRef = C->addNode(newNode);
      // update/create new elements
      elemRef newElem3 = C->addElement(nodes[a],newRef,nodes[c],nodes[d]);
      elemRef newElem4 = C->addElement(newRef,nodes[b],nodes[c],nodes[d]);
      // need to updateFace on elements on acd and bcd faces: acd face
      // still on this element so no changes needed
      if (acd.cid != -1) mesh[acd.cid].updateFace(acd.idx, myRef, newElem3);
      if (bcd.cid != -1) mesh[bcd.cid].updateFace(bcd.idx, myRef, newElem4);
      // this element has the abd face, so c will be the new node
      nodes[c] = newRef;
      // set faces of the two new elements
      C->theElements[newElem3.idx].faceElements[0].cid = abc.cid;
      C->theElements[newElem3.idx].faceElements[0].idx = theResult->ance;
      C->theElements[newElem3.idx].faceElements[1] = myRef;
      C->theElements[newElem3.idx].faceElements[2] = acd;
      C->theElements[newElem3.idx].faceElements[3] = newElem4;
      C->theElements[newElem4.idx].faceElements[0].cid = abc.cid;
      C->theElements[newElem4.idx].faceElements[0].idx = theResult->bnce;
      C->theElements[newElem4.idx].faceElements[1] = myRef;
      C->theElements[newElem4.idx].faceElements[2] = newElem3;
      C->theElements[newElem4.idx].faceElements[3] = bcd;
      // update my faces with the two new elements
      faceElements[a+c+d-3] = newElem3;
      faceElements[b+c+d-3] = newElem4;
      // pass on target volume to the two new elements
      C->theElements[newElem3.idx].setTargetVolume(targetVolume);
      C->theElements[newElem4.idx].setTargetVolume(targetVolume);
      calculateVolume(); // since this element has changed in size
      // tell remote elems about their new faces
      //CkPrintf("      -> Updating remote faces [%d,%d] and [%d,%d]...\n",
      //theResult->ance, abc.cid, theResult->bnce, abc.cid);
      if (abc.cid != -1) {
	mesh[abc.cid].updateFace(theResult->ance, newElem3.cid, newElem3.idx);
	mesh[abc.cid].updateFace(theResult->bnce, newElem4.cid, newElem4.idx);
	/* SMOOTHING
	double smooth = targetVolume / C->smoothness;
	mesh[abc.cid].refineElement(theResult->ance, smooth);
	mesh[abc.cid].refineElement(theResult->bnce, smooth);
	mesh[abc.cid].refineElement(abc.idx, smooth);
	*/
      }
    }
    else if (theResult->success == 0) { // wait for split
      //CkPrintf("    -> Refine: %d on %d: Large face (%d,%d) split failed.\n",
      //       myRef.idx, myRef.cid, abc.idx, abc.cid);
      mesh[myRef.cid].refineElement(myRef.idx, currentVolume);
    }
    else CkPrintf("Refine: %d on %d: WHAT THE FRELL?\n", myRef.idx, myRef.cid);
  }
  if (bcd.cid != -1) mesh[bcd.cid].unlockChunk(myRef.cid);
  if (acd.cid != -1) mesh[acd.cid].unlockChunk(myRef.cid);
  if (abc.cid != -1) mesh[abc.cid].unlockChunk(myRef.cid);
  C->unlockLocalChunk(myRef.cid);
  //CkPrintf("Refine: DONE %d on %d\n", myRef.idx, myRef.cid);
}

int element::lockLF(node n1, node n2, node n3, node n4, elemRef requester, 
			double prio)
{
  int a=-1, b=-1, c=-1, d=-1;
  elemRef abc, acd, bcd;
  // first try to lock this chunk
  intMsg *result;
  int iResult = C->lockLocalChunk(requester.cid, prio);
  while (iResult != 1) {
    if (iResult == 0) return 0;
    else if (iResult == -1) CthYield();
    iResult = C->lockLocalChunk(requester.cid, prio);
  }
  // align nodes with requester
  int found = 0;
  for (int i=0; i<4; i++)
    if (faceElements[i] == requester)
      found = 1;
  CmiAssert(found);
  a = getNode(n1);
  b = getNode(n2);
  c = getNode(n3);
  CmiAssert((a != -1) && (b != -1) && (c != -1));
  // double check abc and get the faces that will need to be locked
  d = 6-a-b-c;
  abc = faceElements[a+b+c-3];
  CmiAssert(abc == requester);
  acd = faceElements[a+c+d-3];
  bcd = faceElements[b+c+d-3];
  // build local locking cloud
  if (acd.cid != -1) {
    result = mesh[acd.cid].lockChunk(requester.cid, prio);
    while (result->anInt != 1) {
      if (result->anInt == 0) {
	C->unlockLocalChunk(requester.cid);
	CkFreeMsg(result);
	return 0;
      }
      else if (result->anInt == -1) {
	CthYield();
	CkFreeMsg(result);
	result = mesh[acd.cid].lockChunk(requester.cid, prio);
      }
    }
    CkFreeMsg(result);
  }
  if (bcd.cid != -1) {
    result = mesh[bcd.cid].lockChunk(requester.cid, prio);
    while (result->anInt != 1) {
      if (result->anInt == 0) {
	C->unlockLocalChunk(requester.cid);
	if (acd.cid != -1) mesh[acd.cid].unlockChunk(requester.cid);
	CkFreeMsg(result);
	return 0;
      }
      else if (result->anInt == -1) {
	CthYield();
	CkFreeMsg(result);
	result = mesh[bcd.cid].lockChunk(requester.cid, prio);
      }
    }
    CkFreeMsg(result);
  }
  return 1;
}

splitResponse *element::splitLF(node in1, node in2, node in3, node in4, elemRef requester)
{ // split neighbor and store data in splitResponse *theResult to transmit 
  // back to initiating tet
  splitResponse *theResult = new splitResponse;
  int a=-1, b=-1, c=-1, d=-1;
  node centerpoint;
  elemRef abc, acd, bcd;

  //CkPrintf("...... Split: %d on %d\n", myRef.idx, myRef.cid);

  theResult->success = 1;
  int found = 0;
  // align nodes with requester
  for (int i=0; i<4; i++)
    if (faceElements[i] == requester)
      found = 1;
  CmiAssert(found);
  a = getNode(in1);
  b = getNode(in2);
  c = getNode(in3);
  CmiAssert((a != -1) && (b != -1) && (c != -1));
  d = 6-a-b-c;
  abc = faceElements[a+b+c-3];
  CmiAssert(abc == requester);
  acd = faceElements[a+c+d-3];
  bcd = faceElements[b+c+d-3];
  // find new node in face
  centerpoint.set((C->theNodes[nodes[a].idx].getCoord(0) + 
		   C->theNodes[nodes[b].idx].getCoord(0) +
		   C->theNodes[nodes[c].idx].getCoord(0)) / 3.0,
		  (C->theNodes[nodes[a].idx].getCoord(1) +
		   C->theNodes[nodes[b].idx].getCoord(1) +
		   C->theNodes[nodes[c].idx].getCoord(1)) / 3.0,
		  (C->theNodes[nodes[a].idx].getCoord(2) + 
		   C->theNodes[nodes[b].idx].getCoord(2) +
		   C->theNodes[nodes[c].idx].getCoord(2)) / 3.0);
  centerpoint.notSurface();    
  centerpoint.notFixed();
  nodeRef newRef = C->addNode(centerpoint);
  // modify this tet and add two new ones
  elemRef newElem1 = C->addElement(nodes[a], newRef, nodes[c], nodes[d]);
  elemRef newElem2 = C->addElement(newRef, nodes[b], nodes[c], nodes[d]);
  // need to updateFace on elements on acd and bcd faces: abd face
  // still on this element so no changes needed
  if (acd.cid != -1) mesh[acd.cid].updateFace(acd.idx, myRef, newElem1);
  if (bcd.cid != -1) mesh[bcd.cid].updateFace(bcd.idx, myRef, newElem2);
  // update this element's c node
  nodes[c] = newRef;
  // set faces of the two new elements
  C->theElements[newElem1.idx].faceElements[1] = myRef;
  C->theElements[newElem1.idx].faceElements[2] = acd;
  C->theElements[newElem1.idx].faceElements[3] = newElem2;
  C->theElements[newElem2.idx].faceElements[1] = myRef;
  C->theElements[newElem2.idx].faceElements[2] = newElem1;
  C->theElements[newElem2.idx].faceElements[3] = bcd;
  // update this elements faces with the two new elements
  faceElements[a+c+d-3] = newElem1;
  faceElements[b+c+d-3] = newElem2;
  // pass on the target volume to the two new elements
  C->theElements[newElem1.idx].setTargetVolume(targetVolume);
  C->theElements[newElem2.idx].setTargetVolume(targetVolume);
  calculateVolume(); // since this element has changed in size
  // place all relevant information in retrun message
  theResult->ance = newElem1.idx;
  theResult->bnce = newElem2.idx;
  theResult->newNode[0] = centerpoint.getCoord(0);
  theResult->newNode[1] = centerpoint.getCoord(1);
  theResult->newNode[2] = centerpoint.getCoord(2);
  // done: unlockChunks
  if (bcd.cid != -1) mesh[bcd.cid].unlockChunk(requester.cid);
  if (acd.cid != -1) mesh[acd.cid].unlockChunk(requester.cid);
  // do this on originator instead
  // C->unlockLocalChunk(requester.cid);
  return theResult;
}

// Longest edge methods
void element::refineLE()
{
  int a, b, c, d, iResult;  // a,b will be the longest edge
  elemRef abc, abd, bcd;
  nodeRef oldNode;
  intMsg *result;
  double length;
  int arc1=0, arc2=0;

  //printf("1: Refine: %d on %d: volume=%lf target=%lf\n", 
  //myRef.idx, myRef.cid, currentVolume, targetVolume);

  length = findLongestEdge(&a, &b, &c, &d);
  iResult = C->lockLocalChunk(myRef.cid, length);
  while (iResult != 1) {
    if (iResult == 0) return;
    else if (iResult == -1) CthYield();
    length = findLongestEdge(&a, &b, &c, &d);
    iResult = C->lockLocalChunk(myRef.cid, length);
  }

  // get the two neighboring elements on longest edge
  abc = getFace(a, b, c); CmiAssert(abc == faceElements[a+b+c-3]);
  abd = getFace(a, b, d); CmiAssert(abd == faceElements[a+b+d-3]);
  bcd = getFace(b, c, d); CmiAssert(bcd == faceElements[b+c+d-3]);
  //printf("2: Refine: %d on %d: volume=%lf target=%lf\n", 
  //myRef.idx, myRef.cid, currentVolume, targetVolume);
  // lock the involved chunks
  if (bcd.cid != -1) { // bcd element exists
    result = mesh[bcd.cid].lockChunk(myRef.cid, length);
    while (result->anInt != 1) {
      if (result->anInt == 0) {
	C->unlockLocalChunk(myRef.cid);
	return;
      }
      else if (result->anInt == -1) {
	CthYield();
	bcd = getFace(b, c, d);
	result = mesh[bcd.cid].lockChunk(myRef.cid, length);
      }
    }
    CkFreeMsg(result);
  }
  if (abc.cid != -1) { // abc element exists
    arc1 = 1;
    // first lock abc
    result = mesh[abc.cid].lockChunk(myRef.cid, length);
    while (result->anInt != 1) {
      //printf("Chunk %d trying to lock %d\n", myRef.cid, abc.cid);
      if (result->anInt == 0) {
	//printf("Chunk %d failed to lock %d\n", myRef.cid, abc.cid);
	if (bcd.cid != -1) mesh[bcd.cid].unlockChunk(myRef.cid);
	C->unlockLocalChunk(myRef.cid);
	return;
      }
      else if (result->anInt == -1) {
	CthYield();
	abc = getFace(a, b, c);
	//printf("Chunk %d trying to lock %d\n", myRef.cid, abc.cid);
	result = mesh[abc.cid].lockChunk(myRef.cid, length);
      }
    }
    // abc locked
    CkFreeMsg(result);
    //printf("3: Refine: %d on %d: locking abc arc\n", myRef.idx, myRef.cid);
    lockArcMsg *lm1 = new lockArcMsg;
    lm1->idx = abc.idx;  
    lm1->prioRef = lm1->parentRef = myRef;  
    lm1->destRef = abd;
    lm1->prio = length;
    lm1->a = C->theNodes[nodes[a].idx];  
    lm1->b = C->theNodes[nodes[b].idx];
    lockResult *lr1 = mesh[abc.cid].lockArc(lm1);
    if (lr1->result == 1) {} // all's well
    else if ((lr1->result == -1) && (abd.cid != -1)) { // didn't reach abd
      arc2 = 1;
      // first lock abd
      result = mesh[abd.cid].lockChunk(myRef.cid, length);
      while (result->anInt != 1) {
	if (result->anInt == 0) {
	  mesh[abc.cid].unlockArc1(abc.idx, myRef.cid, myRef, abd, 
				   C->theNodes[nodes[a].idx], 
				   C->theNodes[nodes[b].idx]);
	  if (bcd.cid != -1) mesh[bcd.cid].unlockChunk(myRef.cid);
	  C->unlockLocalChunk(myRef.cid);
	  return;
	}
	else if (result->anInt == -1) {
	  CthYield();
	  abd = getFace(a, b, d);
	  result = mesh[abd.cid].lockChunk(myRef.cid, length);
	}
      }
      // abd locked
      CkFreeMsg(result);
      //printf("3.5: Refine: %d on %d: now abd arc\n", myRef.idx, myRef.cid);
      lockArcMsg *lm2 = new lockArcMsg;
      lm2->idx = abd.idx;  
      lm2->prioRef = lm2->parentRef = myRef;  
      lm2->destRef = abc;
      lm2->prio = length;
      lm2->a = C->theNodes[nodes[a].idx];  
      lm2->b = C->theNodes[nodes[b].idx];
      lockResult *lr2 = mesh[abd.cid].lockArc(lm2);
      if (lr2->result == 0) {
	mesh[abc.cid].unlockArc1(abc.idx, myRef.cid, myRef, abd, 
				 C->theNodes[nodes[a].idx], 
				 C->theNodes[nodes[b].idx]);
	if (bcd.cid != -1) mesh[bcd.cid].unlockChunk(myRef.cid);
	C->unlockLocalChunk(myRef.cid);
	return;
      }
    }
    else if (lr1->result == 0) { // failed
      if (bcd.cid != -1) mesh[bcd.cid].unlockChunk(myRef.cid);
      C->unlockLocalChunk(myRef.cid);
      return;  
    }
  }
  else if (abd.cid != -1) { // abd element exists
    arc2 = 1;
    result = mesh[abd.cid].lockChunk(myRef.cid, length);
    while (result->anInt != 1) {
      if (result->anInt == 0) {
	if (bcd.cid != -1) mesh[bcd.cid].unlockChunk(myRef.cid);
	C->unlockLocalChunk(myRef.cid);
	return;
      }
      else if (result->anInt == -1) {
	CthYield();
	abd = getFace(a, b, d);
	result = mesh[abd.cid].lockChunk(myRef.cid, length);
      }
    }
    CkFreeMsg(result);
    //printf("4: Refine: %d on %d: locking abd arc\n", myRef.idx, myRef.cid);
    lockArcMsg *lm3 = new lockArcMsg;
    lm3->idx = abd.idx;  
    lm3->prioRef = lm3->parentRef = myRef;  
    lm3->destRef = abc;
    lm3->prio = length;
    lm3->a = C->theNodes[nodes[a].idx];  
    lm3->b = C->theNodes[nodes[b].idx];
    lockResult *lr3 = mesh[abd.cid].lockArc(lm3);
    if (lr3->result == 0) { // failed
      if (bcd.cid != -1) mesh[bcd.cid].unlockChunk(myRef.cid);
      C->unlockLocalChunk(myRef.cid);
      return;  
    }
  }

  //printf("5: Refine: %d on %d: volume=%lf target=%lf\n", 
  //   myRef.idx, myRef.cid, currentVolume, targetVolume);

  // build the new node at the midpoint on the longest edge
  node mid = C->theNodes[nodes[a].idx].midpoint(C->theNodes[nodes[b].idx]);
  if (C->edgeOnSurface(nodes[a].idx, nodes[b].idx))
    mid.setSurface();
  else mid.notSurface();
  if (C->theNodes[nodes[a].idx].isFixed() &&
      C->theNodes[nodes[b].idx].isFixed())
    mid.fix();
  else mid.notFixed();
  nodeRef newNode = C->addNode(mid);
  //printf("6: Refine: %d on %d: volume=%lf target=%lf\n", 
  //myRef.idx, myRef.cid, currentVolume, targetVolume);
  // split this element by creating one new one
  elemRef newElem = C->addElement(newNode, nodes[b], nodes[c], nodes[d]);
  // update faces and nodes of the two elements
  C->theElements[newElem.idx].faceElements[0] = faceElements[a+b+c-3];
  C->theElements[newElem.idx].faceElements[1] = faceElements[a+b+d-3];
  C->theElements[newElem.idx].faceElements[2] = myRef;
  C->theElements[newElem.idx].faceElements[3] = faceElements[b+c+d-3];
  C->theElements[newElem.idx].calculateVolume();
  C->theElements[newElem.idx].setTargetVolume(targetVolume);
  if (faceElements[b+c+d-3].cid != -1)
    mesh[faceElements[b+c+d-3].cid].updateFace(faceElements[b+c+d-3].idx, 
					       myRef, newElem);
  faceElements[b+c+d-3] = newElem;
  oldNode = nodes[b]; 
  nodes[b] = newNode;
  calculateVolume();
  //printf("7: Refine: %d on %d: volume=%lf target=%lf\n", 
  //   myRef.idx, myRef.cid, currentVolume, targetVolume);
  // split the neighbor
  if (abc.cid != -1) {
    LEsplitMsg *lsm1 = new LEsplitMsg;
    lsm1->idx = abc.idx;
    lsm1->root = myRef;
    lsm1->parent = myRef;
    lsm1->newNodeRef = newNode;
    lsm1->newNode = C->theNodes[newNode.idx];
    lsm1->newRootElem = newElem;
    lsm1->newElem = newElem;
    lsm1->targetElem = abd;
    lsm1->targetVol = targetVolume;
    lsm1->a = C->theNodes[nodes[a].idx];
    lsm1->b = C->theNodes[oldNode.idx];
    LEsplitResult *lsr1 = mesh[abc.cid].LEsplit(lsm1);
    C->theElements[newElem.idx].faceElements[0] = lsr1->newElem1;
    if (lsr1->status == 0) {
      if (abd.cid != -1) {
	LEsplitMsg *lsm2 = new LEsplitMsg;
	lsm2->idx = abd.idx;
	lsm2->root = myRef;
	lsm2->parent = myRef;
	lsm2->newNodeRef = newNode;
	lsm2->newNode = C->theNodes[newNode.idx];
	lsm2->newRootElem = newElem;
	lsm2->newElem = newElem;
	lsm2->targetElem = abc;
	lsm2->targetVol = targetVolume;
	lsm2->a = C->theNodes[nodes[a].idx];
	lsm2->b = C->theNodes[oldNode.idx];
	LEsplitResult *lsr2 = mesh[abd.cid].LEsplit(lsm2);
	C->theElements[newElem.idx].faceElements[1] = lsr2->newElem1;
      }
    }
    else {
      CmiAssert(lsr1->status == 1);
      C->theElements[newElem.idx].faceElements[1] = lsr1->newElem2;
    }
  }
  else if (abd.cid != -1) {
    LEsplitMsg *lsm3 = new LEsplitMsg;
    lsm3->idx = abd.idx;
    lsm3->root = myRef;
    lsm3->parent = myRef;
    lsm3->newNodeRef = newNode;
    lsm3->newNode = C->theNodes[newNode.idx];
    lsm3->newRootElem = newElem;
    lsm3->newElem = newElem;
    lsm3->targetElem = abc;
    lsm3->targetVol = targetVolume;
    lsm3->a = C->theNodes[nodes[a].idx];
    lsm3->b = C->theNodes[oldNode.idx];
    LEsplitResult *lsr3 = mesh[abd.cid].LEsplit(lsm3);
    CmiAssert(lsr3->status == 0);
    C->theElements[newElem.idx].faceElements[1] = lsr3->newElem1;
  }
  //printf("8: Refine: %d on %d: volume=%lf target=%lf\n", 
  //   myRef.idx, myRef.cid, currentVolume, targetVolume);
  if ((arc1 == 1) && (abc.cid != -1))
    mesh[abc.cid].unlockArc2(abc.idx, myRef.cid, myRef, abd, 
			     C->theNodes[nodes[a].idx], 
			     C->theNodes[nodes[b].idx]);
  //printf("9: Refine: %d on %d: volume=%lf target=%lf\n", 
  //   myRef.idx, myRef.cid, currentVolume, targetVolume);
  if ((arc2 == 1) && (abd.cid != -1))
    mesh[abd.cid].unlockArc2(abd.idx, myRef.cid, myRef, abc, 
			     C->theNodes[nodes[a].idx], 
			     C->theNodes[nodes[b].idx]);
  //printf("10: Refine: %d on %d: volume=%lf target=%lf\n", 
  //   myRef.idx, myRef.cid, currentVolume, targetVolume);
  if (bcd.cid != -1)
    mesh[bcd.cid].unlockChunk(myRef.cid);
  //printf("11: Refine: %d on %d: volume=%lf target=%lf\n", 
  //   myRef.idx, myRef.cid, currentVolume, targetVolume);
  C->unlockLocalChunk(myRef.cid);
}

LEsplitResult *element::LEsplit(elemRef root, elemRef parent, 
				nodeRef newNodeRef, node newNode, 
 				elemRef newRootElem, elemRef newElem, 
				elemRef targetElem, double targetVol, 
				node aIn, node bIn)
{
  int a=-1, b=-1, c=-1, d=-1;  // a,b will be the longest edge
  elemRef abc, abd; // abc will be parent
  nodeRef oldNode; 

  calculateVolume();
  //printf("1: LEsplit: O[%d,%d] from [%d,%d] on [%d,%d] dest [%d,%d] vol=%lf->%lf  \n", 
  //   root.idx, root.cid, parent.idx, parent.cid, myRef.idx, myRef.cid, targetElem.idx, targetElem.cid,
  //   currentVolume, targetVolume);

  // find ab edge
  for (int i=0; i<4; i++) {
    if (C->theNodes[nodes[i].idx] == aIn)
      a = i;
    else if (C->theNodes[nodes[i].idx] == bIn)
      b = i;
    else if (c == -1)
      c = i;
    else
      d = i;
  }

  CmiAssert((a!=-1)&&(b!=-1)&&(c!=-1)&&(d!=-1));
  // get the two neighboring elements on longest edge
  abc = getFace(a, b, c);
  if (!(abc == parent)) {
    int tmp = c;
    c = d; 
    d = tmp;
    abd = abc; 
    abc = getFace(a, b, c);
  }
  else abd = getFace(a, b, d);
  CmiAssert(abc == parent);
  //printf("2: LEsplit: O[%d,%d] from [%d,%d] on [%d,%d] dest [%d,%d] vol=%lf->%lf  \n", 
  //   root.idx, root.cid, parent.idx, parent.cid, myRef.idx, myRef.cid, targetElem.idx, targetElem.cid,
  //   currentVolume, targetVolume);
  // add the new node locally if necessary
  if (newNodeRef.cid != myRef.cid)
    newNodeRef = C->addNode(newNode);

  // split this element by creating one new one
  elemRef myNewElem = C->addElement(newNodeRef, nodes[b], nodes[c], nodes[d]);
  // update faces and nodes of the two elements
  C->theElements[myNewElem.idx].faceElements[0] = newElem;
  C->theElements[myNewElem.idx].faceElements[1] = faceElements[a+b+d-3];
  C->theElements[myNewElem.idx].faceElements[2] = myRef;
  C->theElements[myNewElem.idx].faceElements[3] = faceElements[b+c+d-3];
  C->theElements[myNewElem.idx].calculateVolume();
  C->theElements[myNewElem.idx].setTargetVolume(targetVolume);
  if (faceElements[b+c+d-3].cid != -1)
    mesh[faceElements[b+c+d-3].cid].updateFace(faceElements[b+c+d-3].idx, 
					       myRef, myNewElem);
  faceElements[b+c+d-3] = myNewElem;
  oldNode = nodes[b];
  nodes[b] = newNodeRef;
  calculateVolume();

  /*
  double smooth = targetVol / C->smoothness;
  mesh[myRef.cid].refineElement(myRef.idx, smooth);
  mesh[myRef.cid].refineElement(myNewElem.idx, smooth);
  */

  //printf("3: LEsplit: O[%d,%d] from [%d,%d] on [%d,%d] dest [%d,%d] vol=%lf->%lf  \n", 
  //   root.idx, root.cid, parent.idx, parent.cid, myRef.idx, myRef.cid, targetElem.idx, targetElem.cid,
  //   currentVolume, targetVolume);
  // split the neighbor
  LEsplitResult *result = new LEsplitResult;
  if (myRef == targetElem) { // finished entire loop
    result->status = 1;
    C->theElements[myNewElem.idx].faceElements[1] = newRootElem;
    result->newElem1 = myNewElem;
    result->newElem2 = myNewElem;
  }
  else if (abd.cid != -1) { // continue path
    LEsplitMsg *lsm = new LEsplitMsg;
    lsm->idx = abd.idx;
    lsm->root = root;
    lsm->parent = myRef;
    lsm->newNodeRef = newNodeRef;
    lsm->newNode = C->theNodes[newNodeRef.idx];
    lsm->newRootElem = newRootElem;
    lsm->newElem = myNewElem;
    lsm->targetElem = targetElem;
    lsm->targetVol = targetVol;
    lsm->a = aIn;
    lsm->b = bIn;
    LEsplitResult *lsr = mesh[abd.cid].LEsplit(lsm);
    C->theElements[myNewElem.idx].faceElements[1] = lsr->newElem1;
    result->newElem1 = myNewElem;
    result->newElem2 = lsr->newElem2;
    result->status = lsr->status;
  }
  else { // dead end
    result->status = 0;
    result->newElem1 = myNewElem;
    result->newElem2.cid = -1; result->newElem2.idx = -1;
  }
  //printf("4: LEsplit: O[%d,%d] from [%d,%d] on [%d,%d] dest [%d,%d] vol=%lf->%lf  \n", 
  //   root.idx, root.cid, parent.idx, parent.cid, myRef.idx, myRef.cid, targetElem.idx, targetElem.cid,
  //   currentVolume, targetVolume);
  return result;
}

lockResult *element::lockArc(elemRef prioRef, elemRef parentRef, double prio,
			     elemRef destRef, node aNode, node bNode)
{ // this element should be locked already
  lockResult *myResult = new lockResult;
  elemRef abc, bcd;
  int a=-1, b=-1, c=-1, d=-1;

  //printf("---> 1:LockArc: O[%d,%d] on [%d,%d] dest [%d,%d]\n", prioRef.idx, 
  //   prioRef.cid, myRef.idx, myRef.cid, destRef.idx, destRef.cid);

  // find ab edge
  for (int i=0; i<4; i++) {
    if (C->theNodes[nodes[i].idx] == aNode)  a = i;
    else if (C->theNodes[nodes[i].idx] == bNode)  b = i;
    else if (c == -1)  c = i;
    else  d = i;
  }
  int ap, bp, cp, dp;
  double length = findLongestEdge(&ap, &bp, &cp, &dp);
  if (!(((ap == a) && (bp == b)) || ((ap == b) && (bp == a)) ||
	(length == prio))) {
    mesh[myRef.cid].refineElement(myRef.idx, getVolume()/2.0);
    myResult->result = 0;
    CkPrintf("Nbr failed LE test...\n");
    C->unlockLocalChunk(prioRef.cid);
    return myResult;
  }

  CmiAssert((a!=-1)&&(b!=-1)&&(c!=-1)&&(d!=-1));
  //printf("---> 2:LockArc: O[%d,%d] on [%d,%d] dest [%d,%d]\n", prioRef.idx, 
  //   prioRef.cid, myRef.idx, myRef.cid, destRef.idx, destRef.cid);
  // get the two neighboring elements on longest edge
  abc = getFace(a, b, c);
  if (abc == parentRef) {
    int tmp = c;
    c = d; 
    d = tmp;
    abc = getFace(a, b, c);
  }
  CmiAssert(!(abc == parentRef));
  //printf("---> 3:LockArc: O[%d,%d] on [%d,%d] dest [%d,%d]\n", prioRef.idx, 
  //   prioRef.cid, myRef.idx, myRef.cid, destRef.idx, destRef.cid);
  bcd = getFace(b, c, d);
  if (bcd.cid != -1) { // bcd element exists
    intMsg *result = mesh[bcd.cid].lockChunk(prioRef.cid, prio);
    //printf("---> 4:LockArc: O[%d,%d] on [%d,%d] dest [%d,%d]\n", prioRef.idx,
    //     prioRef.cid, myRef.idx, myRef.cid, destRef.idx, destRef.cid);
    while (result->anInt != 1) {
      if (result->anInt == 0) {
	myResult->result = 0;
	C->unlockLocalChunk(prioRef.cid);
	return myResult;
      }
      else if (result->anInt == -1) {
	CthYield();
	bcd = getFace(b, c, d);
	result = mesh[bcd.cid].lockChunk(prioRef.cid, prio);
      }
    }
  }
  //printf("---> 5:LockArc: O[%d,%d] on [%d,%d] dest [%d,%d]\n", prioRef.idx, 
  //   prioRef.cid, myRef.idx, myRef.cid, destRef.idx, destRef.cid);
  // continue locking element arc
  if (myRef == destRef)  {
    //printf("---> 6:LockArc: O[%d,%d] on [%d,%d] dest [%d,%d]: AT DEST: SUCCEED\n", 
    //     prioRef.idx, prioRef.cid, myRef.idx, myRef.cid, destRef.idx, destRef.cid);    
    myResult->result = 1;
  }
  else if (abc.cid != -1) {
    intMsg *result = mesh[abc.cid].lockChunk(prioRef.cid, prio);
    //printf("---> 7a:LockArc: O[%d,%d] on [%d,%d] dest [%d,%d]: trying next(%d,%d)\n", 
    //     prioRef.idx, prioRef.cid, myRef.idx, myRef.cid, 
    //     destRef.idx, destRef.cid, abc.idx, abc.cid);
    while (result->anInt != 1) {
      if (result->anInt == 0) {
	myResult->result = 0;
	if (bcd.cid != -1) mesh[bcd.cid].unlockChunk(prioRef.cid);
	C->unlockLocalChunk(prioRef.cid);
	return myResult;
      }
      else if (result->anInt == -1) {
	CthYield();
	abc = getFace(a, b, c);
	result = mesh[abc.cid].lockChunk(prioRef.cid, prio);
      }
    }
    //printf("---> 7b:LockArc: O[%d,%d] on [%d,%d] dest [%d,%d]: NEXT(%d,%d)\n", 
    //     prioRef.idx, prioRef.cid, myRef.idx, myRef.cid, destRef.idx, destRef.cid, abc.idx, abc.cid);
    lockArcMsg *lm1 = new lockArcMsg;
    lm1->idx = abc.idx;  
    lm1->prioRef = prioRef;
    lm1->parentRef = myRef;  
    lm1->destRef = destRef;
    lm1->prio = prio;
    lm1->a = aNode;
    lm1->b = bNode;
    lockResult *lr1 = mesh[abc.cid].lockArc(lm1);
    //printf("---> 7c:LockArc: O[%d,%d] on [%d,%d] dest [%d,%d]: from(%d,%d) result=%d\n", 
    //prioRef.idx, prioRef.cid, myRef.idx, myRef.cid, destRef.idx, destRef.cid, abc.idx, abc.cid, lr1->result);
    if (lr1->result == 1)  myResult->result = 1;
    else if (lr1->result == -1)  myResult->result = -1;
    else if (lr1->result == 0)  { 	
      if (bcd.cid != -1) mesh[bcd.cid].unlockChunk(prioRef.cid);
      C->unlockLocalChunk(prioRef.cid);
      myResult->result = 0;
    }
  }
  else {
    //printf("---> 8:LockArc: O[%d,%d] on [%d,%d] dest [%d,%d]: NOT AT DEST: DONE\n", 
    //     prioRef.idx, prioRef.cid, myRef.idx, myRef.cid, destRef.idx, destRef.cid);    
    myResult->result = -1;
  }
  return myResult;
}

void element::unlockArc1(int prio, elemRef parentRef, elemRef destRef, 
			 node aNode, node bNode)
{
  elemRef abc, bcd;
  int a=-1, b=-1, c=-1, d=-1;
  // unlock this element
  //printf("---> 1:UnlockArc1: on [%d,%d] dest [%d,%d]\n", 
  //   myRef.idx, myRef.cid, destRef.idx, destRef.cid);    
  C->unlockLocalChunk(prio);
  // find ab edge
  for (int i=0; i<4; i++) {
    if (C->theNodes[nodes[i].idx] == aNode)  a = i;
    else if (C->theNodes[nodes[i].idx] == bNode)  b = i;
    else if (c == -1)  c = i;
    else  d = i;
  }
  CmiAssert((a!=-1)&&(b!=-1)&&(c!=-1)&&(d!=-1));
  // get the next element to unlock
  //printf("---> 2:UnlockArc1: on [%d,%d] dest [%d,%d]\n", 
  //   myRef.idx, myRef.cid, destRef.idx, destRef.cid);    
  abc = getFace(a, b, c);
  if (abc == parentRef) {
    int tmp = c;
    c = d; 
    d = tmp;
    abc = getFace(a, b, c);
  }
  CmiAssert(!(abc == parentRef));
  bcd = getFace(b, c, d);
  // continue unlocking element arc
  //printf("---> 3:UnlockArc1: on [%d,%d] dest [%d,%d]\n", 
  //   myRef.idx, myRef.cid, destRef.idx, destRef.cid);    
  if (bcd.cid != -1)
    mesh[bcd.cid].unlockChunk(prio);
  //printf("---> 4:UnlockArc1: on [%d,%d] dest [%d,%d]\n", 
  //   myRef.idx, myRef.cid, destRef.idx, destRef.cid);    
  if (myRef == destRef) return;
  else if (abc.cid != -1)
    mesh[abc.cid].unlockArc1(abc.idx, prio, myRef, destRef, aNode, bNode);
  //printf("---> 5:UnlockArc1: on [%d,%d] dest [%d,%d]\n", 
  //   myRef.idx, myRef.cid, destRef.idx, destRef.cid);    
}

void element::unlockArc2(int prio, elemRef parentRef, elemRef destRef, 
			 node aNode, node bNode)
{
  elemRef abc, bcd;
  int a=-1, b=-1, c=-1, d=-1;
  // unlock this element
  //printf("---> 1:UnlockArc2: on [%d,%d] dest [%d,%d]\n", 
  //   myRef.idx, myRef.cid, destRef.idx, destRef.cid);    
  C->unlockLocalChunk(prio);
  // find ab edge
  for (int i=0; i<4; i++) {
    if (C->theNodes[nodes[i].idx] == aNode)  a = i;
    else if (C->theNodes[nodes[i].idx] == bNode)  b = i;
    else if (c == -1)  c = i;
    else  d = i;
  }
  CmiAssert((a!=-1)&&(b!=-1)&&(c!=-1)&&(d!=-1));
  // get the next element to unlock
  //printf("---> 2:UnlockArc2: on [%d,%d] dest [%d,%d]\n", 
  //   myRef.idx, myRef.cid, destRef.idx, destRef.cid);    
  abc = getFace(a, b, c);
  if (abc == parentRef) {
    int tmp = c;
    c = d; 
    d = tmp;
    abc = getFace(a, b, c);
  }
  CmiAssert(!(abc == parentRef));
  //printf("---> 3:UnlockArc2: on [%d,%d] dest [%d,%d]\n", 
  //   myRef.idx, myRef.cid, destRef.idx, destRef.cid);    
  bcd = getFace(b, c, d);
  // continue unlocking element arc
  elemRef bcdNbr = C->theElements[bcd.idx].faceElements[3];
  if (bcdNbr.cid != -1)
    mesh[bcdNbr.cid].unlockChunk(prio);
  //printf("---> 4:UnlockArc2: on [%d,%d] dest [%d,%d]\n", 
  //   myRef.idx, myRef.cid, destRef.idx, destRef.cid);    
  if (myRef == destRef) return;
  else if (abc.cid != -1)
    mesh[abc.cid].unlockArc2(abc.idx, prio, myRef, destRef, aNode, bNode);
  //printf("---> 5:UnlockArc2: on [%d,%d] dest [%d,%d]\n", 
  //   myRef.idx, myRef.cid, destRef.idx, destRef.cid);    
}

void element::refineCP()
{
  int a, b, c, d, iResult;
  node n1, n2, n3, n4, centerpoint;
  elemRef abcn, ancd, nbcd, abnd;
  intMsg *result;
  double length;

  //CkPrintf("Refine: %d on %d: volume=%lf target=%lf\n",
  //   myRef.idx, myRef.cid, currentVolume, targetVolume);
  if ((currentVolume == 0.0) || (targetVolume == 0.0)) {
    CkPrintf("WARNING: %d on %d has dwindled; using bailout!\n", 
	     myRef.idx, myRef.cid);
    targetVolume = -1.0;
    return;
  }
  if (currentVolume < targetVolume) return;

  // build local locking cloud
  length = findLongestEdge(&a, &b, &c, &d);
  iResult = C->lockLocalChunk(myRef.cid, length);
  while (iResult != 1) {
    if (iResult == 0) return;
    else if (iResult == -1) CthYield();
    length = findLongestEdge(&a, &b, &c, &d);
    iResult = C->lockLocalChunk(myRef.cid, length);
  }
  a = 0;
  b = 1;
  c = 2;
  d = 3;
  abcn = faceElements[0];
  abnd = faceElements[1];
  ancd = faceElements[2];
  nbcd = faceElements[3];
  if (abnd.cid != -1) {
    result = mesh[abnd.cid].lockChunk(myRef.cid, length);
    while (result->anInt != 1) {
      if (result->anInt == 0) {
	C->unlockLocalChunk(myRef.cid);
	CkFreeMsg(result);
	return;
      }
      else if (result->anInt == -1) {
	CthYield();
	CkFreeMsg(result);
	result = mesh[abnd.cid].lockChunk(myRef.cid, length);
      }
    }
    CkFreeMsg(result);
  }
  if (ancd.cid != -1) {
    result = mesh[ancd.cid].lockChunk(myRef.cid, length);
    while (result->anInt != 1) {
      if (result->anInt == 0) {
	C->unlockLocalChunk(myRef.cid);
	if (abnd.cid != -1) mesh[abnd.cid].unlockChunk(myRef.cid);
	CkFreeMsg(result);
	return;
      }
      else if (result->anInt == -1) {
	CthYield();
	CkFreeMsg(result);
	result = mesh[ancd.cid].lockChunk(myRef.cid, length);
      }
    }
    CkFreeMsg(result);
  }
  if (nbcd.cid != -1) {
    result = mesh[nbcd.cid].lockChunk(myRef.cid, length);
    while (result->anInt != 1) {
      if (result->anInt == 0) {
	C->unlockLocalChunk(myRef.cid);
	if (abnd.cid != -1) mesh[abnd.cid].unlockChunk(myRef.cid);
	if (ancd.cid != -1) mesh[ancd.cid].unlockChunk(myRef.cid);
	CkFreeMsg(result);
	return;
      }
      else if (result->anInt == -1) {
	CthYield();
	CkFreeMsg(result);
	result = mesh[nbcd.cid].lockChunk(myRef.cid, length);
      }
    }
    CkFreeMsg(result);
  }

  // get nodeRefs for nodes
  //  CkPrintf("  -> Refine: %d on %d: Local cloud built.\n",myRef.idx,myRef.cid);
  n1 = C->theNodes[nodes[a].idx];
  n2 = C->theNodes[nodes[b].idx];
  n3 = C->theNodes[nodes[c].idx];
  n4 = C->theNodes[nodes[d].idx];
  // add new node in center of tet and split
  centerpoint.set(
    (n1.getCoord(0)+n2.getCoord(0)+n3.getCoord(0)+n4.getCoord(0))/4.0,
    (n1.getCoord(1)+n2.getCoord(1)+n3.getCoord(1)+n4.getCoord(1))/4.0,
    (n1.getCoord(2)+n2.getCoord(2)+n3.getCoord(2)+n4.getCoord(2))/4.0);
  centerpoint.notSurface();    
  centerpoint.notFixed();
  nodeRef newNode = C->addNode(centerpoint);
  /*  CkPrintf("   -> Refine: %d on %d: *** New node: (%lf,%lf,%lf) ***\n",
	   myRef.idx, myRef.cid, centerpoint.getCoord(0), 
	   centerpoint.getCoord(1), centerpoint.getCoord(2));
  */
  // modify this tet and add three new ones
  elemRef newElem1 = C->addElement(nodes[a],nodes[b],newNode,nodes[d]);
  elemRef newElem2 = C->addElement(nodes[a],newNode,nodes[c],nodes[d]);
  elemRef newElem3 = C->addElement(newNode,nodes[b],nodes[c],nodes[d]);
  // need to updateFace on elements on abd acd and bcd faces: abc face
  // still on this element so no changes needed
  if (abnd.cid != -1)  mesh[abnd.cid].updateFace(abnd.idx, myRef, newElem1);
  if (ancd.cid != -1)  mesh[ancd.cid].updateFace(ancd.idx, myRef, newElem2);
  if (nbcd.cid != -1)  mesh[nbcd.cid].updateFace(nbcd.idx, myRef, newElem3);
  // set faces for the new elements
  C->theElements[newElem1.idx].faceElements[0] = myRef;
  C->theElements[newElem1.idx].faceElements[1] = abnd;
  C->theElements[newElem1.idx].faceElements[2] = newElem2;
  C->theElements[newElem1.idx].faceElements[3] = newElem3;
  C->theElements[newElem2.idx].faceElements[0] = myRef;
  C->theElements[newElem2.idx].faceElements[1] = newElem1;
  C->theElements[newElem2.idx].faceElements[2] = ancd;
  C->theElements[newElem2.idx].faceElements[3] = newElem3;
  C->theElements[newElem3.idx].faceElements[0] = myRef;
  C->theElements[newElem3.idx].faceElements[1] = newElem1;
  C->theElements[newElem3.idx].faceElements[2] = newElem2;
  C->theElements[newElem3.idx].faceElements[3] = nbcd;
  // update my faces with the three new elements
  faceElements[a+b+d-3] = newElem1;
  faceElements[a+c+d-3] = newElem2;
  faceElements[b+c+d-3] = newElem3;
  // pass target volume on to the three new elements
  C->theElements[newElem1.idx].setTargetVolume(targetVolume);
  C->theElements[newElem2.idx].setTargetVolume(targetVolume);
  C->theElements[newElem3.idx].setTargetVolume(targetVolume);
  // update local nodes with new node
  nodes[d] = newNode;
  calculateVolume(); // since this element has changed in size
  if (abnd.cid != -1) mesh[abnd.cid].unlockChunk(myRef.cid);
  if (ancd.cid != -1) mesh[ancd.cid].unlockChunk(myRef.cid);
  if (nbcd.cid != -1) mesh[nbcd.cid].unlockChunk(myRef.cid);
  C->unlockLocalChunk(myRef.cid);
  //CkPrintf("Refine: DONE %d on %d\n", myRef.idx, myRef.cid);
}

void element::coarsen()
{
}

// Mesh improvement operations
void element::improveElement()
{
  for (int i=0; i<4; i++)
    if (!C->theNodes[nodes[i].idx].isFixed()) {
      if (C->theNodes[nodes[i].idx].onSurface())
	improveSurfaceNode(i);
      else improveInternalNode(i);
    }
}

void element::improveInternalNode(int n)
{
  int ot1=(n+1)%4, ot2=(n+2)%4, ot3=(n+3)%4;
  node aNode=C->theNodes[nodes[n].idx], n1=C->theNodes[nodes[ot1].idx],
    n2=C->theNodes[nodes[ot2].idx], n3=C->theNodes[nodes[ot3].idx];
  double l1=aNode.distance(n1), l2=aNode.distance(n2), l3=aNode.distance(n3),
    l4=n1.distance(n2), l5=n2.distance(n3), l6=n3.distance(n1);
  double r = (l1 + l2 + l3 + l4 + l5 + l6) / 6.0, s1, s2, s3;
  s1 = l1 - r;  s2 = l2 - r; s3 = l3 - r;
  node v1, v2, v3, vote;
  if (l1 > r)
    v1.set( (r*aNode.getCoord(0) + s1*n1.getCoord(0)) / (r+s1),
	    (r*aNode.getCoord(1) + s1*n1.getCoord(1)) / (r+s1),
	    (r*aNode.getCoord(2) + s1*n1.getCoord(2)) / (r+s1) );
  else v1 = n1;
  if (l2 > r)
    v2.set( (r*aNode.getCoord(0) + s2*n2.getCoord(0)) / (r+s2),
	    (r*aNode.getCoord(1) + s2*n2.getCoord(1)) / (r+s2),
	    (r*aNode.getCoord(2) + s2*n2.getCoord(2)) / (r+s2) );
  else v2 = n2;
  if (l3 > r)
    v3.set( (r*aNode.getCoord(0) + s3*n3.getCoord(0)) / (r+s3),
	    (r*aNode.getCoord(1) + s3*n3.getCoord(1)) / (r+s3),
	    (r*aNode.getCoord(2) + s3*n3.getCoord(2)) / (r+s3) );
  else v3 = n3;
  vote.set( (v1.getCoord(0) + v2.getCoord(0) + v3.getCoord(0)) / 3.0, 
	    (v1.getCoord(1) + v2.getCoord(1) + v3.getCoord(1)) / 3.0, 
	    (v1.getCoord(2) + v2.getCoord(2) + v3.getCoord(2)) / 3.0 );
  nodeVoteMsg *nvm = new nodeVoteMsg;
  CkPrintf("Node (%lf,%lf,%lf) has vote to move to (%lf,%lf,%lf)\n",
	   aNode.getCoord(0), aNode.getCoord(1), aNode.getCoord(2),
	   vote.getCoord(0), vote.getCoord(1), vote.getCoord(2));
  nvm->oldCoord[0] = aNode.getCoord(0);
  nvm->oldCoord[1] = aNode.getCoord(1);
  nvm->oldCoord[2] = aNode.getCoord(2);
  nvm->newCoord[0] = vote.getCoord(0);
  nvm->newCoord[1] = vote.getCoord(1);
  nvm->newCoord[2] = vote.getCoord(2);
  mesh.relocationVote(nvm);
}

void element::improveSurfaceNode(int n)
{
  int ot1=(n+1)%4, ot2=(n+2)%4, ot3=(n+3)%4;
  if (C->faceOnSurface(nodes[n].idx, nodes[ot1].idx, nodes[ot2].idx))
    improveSurfaceNodeHelp(n, ot1, ot2);
  else if (C->faceOnSurface(nodes[n].idx, nodes[ot1].idx, nodes[ot3].idx))
    improveSurfaceNodeHelp(n, ot1, ot3);
  else if (C->faceOnSurface(nodes[n].idx, nodes[ot2].idx, nodes[ot3].idx))
    improveSurfaceNodeHelp(n, ot2, ot3);
}

void element::improveSurfaceNodeHelp(int n, int ot1, int ot2)
{
  node aNode=C->theNodes[nodes[n].idx], n1=C->theNodes[nodes[ot1].idx],
    n2=C->theNodes[nodes[ot2].idx];
  double l1=aNode.distance(n1), l2=aNode.distance(n2), l3=n1.distance(n2);
  double r = (l1 + l2 + l3) / 3.0, s1, s2;
  s1 = l1 - r;  s2 = l2 - r;
  node v1, v2, vote;
  if (l1 > r)
    v1.set( (r*aNode.getCoord(0) + s1*n1.getCoord(0)) / (r+s1),
	    (r*aNode.getCoord(1) + s1*n1.getCoord(1)) / (r+s1),
	    (r*aNode.getCoord(2) + s1*n1.getCoord(2)) / (r+s1) );
  else v1 = n1;
  if (l2 > r)
    v2.set( (r*aNode.getCoord(0) + s2*n2.getCoord(0)) / (r+s2),
	    (r*aNode.getCoord(1) + s2*n2.getCoord(1)) / (r+s2),
	    (r*aNode.getCoord(2) + s2*n2.getCoord(2)) / (r+s2) );
  else v2 = n2;
  vote.set( (v1.getCoord(0) + v2.getCoord(0)) / 2.0, 
	    (v1.getCoord(1) + v2.getCoord(1)) / 2.0, 
	    (v1.getCoord(2) + v2.getCoord(2)) / 2.0 );
  nodeVoteMsg *nvm = new nodeVoteMsg;
  nvm->oldCoord[0] = aNode.getCoord(0);
  nvm->oldCoord[1] = aNode.getCoord(1);
  nvm->oldCoord[2] = aNode.getCoord(2);
  nvm->newCoord[0] = vote.getCoord(0);
  nvm->newCoord[1] = vote.getCoord(1);
  nvm->newCoord[2] = vote.getCoord(2);
  mesh.relocationVote(nvm);
}

int element::LEtest() 
{
  double ml = 0.0, ml2 = 0.0, ml3 = 0.0, l[4][4];
  for (int i=0; i<4; i++)
    for (int j=i+1; j<4; j++) {
      l[i][j] = C->theNodes[nodes[i].idx].distance(C->theNodes[nodes[j].idx]);
      if (l[i][j] > ml) {
	ml3 = ml2;
	ml2 = ml;
	ml = l[i][j];
      }
    }
  if (ml3 + (0.33*ml3) <= ml) return 1;
  return 0;
}

int element::LFtest() 
{
  double lf=0.0, lf2=0.0, f[4];
  f[0] = getArea(0,1,2);
  f[1] = getArea(0,1,3);
  f[2] = getArea(0,2,3);
  f[3] = getArea(1,2,3);
  for (int i=0; i<4; i++) if (f[i] > lf) { lf2 = lf; lf = f[i]; }
  if (lf2 + (0.25*lf2) <= lf) return 1;
  return 0;
}

int element::CPtest() 
{ 
  double avg = 0.0, l[4][4];
  for (int i=0; i<4; i++)
    for (int j=i+1; j<4; j++) {
      l[i][j] = C->theNodes[nodes[i].idx].distance(C->theNodes[nodes[j].idx]);
      avg += l[i][j];
    }
  avg = avg / 6.0;
  for (int i=0; i<4; i++)
    for (int j=i+1; j<4; j++) {
      if ((l[i][j] > avg) && (l[i][j] - l[i][j]*0.25 > avg))
	return 0;
      else if ((l[i][j] < avg) && (l[i][j] + l[i][j]*0.25 < avg))
	return 0;
    }
  return 1; 
}
