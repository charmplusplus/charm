#include "barnes.h"

TreePiece::TreePiece(nodeptr p, int which, bool isTopLevel_, int level_){
  isTopLevel = isTopLevel_;
  myLevel = level_;
  if(!isTopLevel_){
    myRoot = makecell(thisIndex);
    ParentOf(myRoot) = p; // this is my parent
    Subp(p)[which] = (nodeptr) myRoot; // i am my parent's 'which' child
    ChildNum(myRoot) = which;
    whichChildAmI = which;
  }
  numTotalMsgs = -1;
  numRecvdMsgs = 0;
  haveChildren = false;
  myNumParticles = 0;

  haveCounts = false;

  for(int i = 0; i < NSUB; i++){
    sentTo[i] = 0;
  }
}

void TreePiece::recvTotalMsgCountsFromPieces(int totalNumFromParent){
  CkAssert(!isTopLevel);
  numTotalMsgs = totalNumFromParent;
  haveCounts = true;
  checkCompletion();
}


void TreePiece::recvTotalMsgCountsFromChunks(CkReductionMsg *msg){
  if(isTopLevel){
    int *data = (int *)msg->getData();
    numTotalMsgs = data[thisIndex]; /* between 0 and numTreePieces, 
                                       which is the number of 
                                       top-level treepieces */
    haveCounts = true;
    checkCompletion();
  }
  delete msg;
}

void TreePiece::checkCompletion(){
  if(numRecvdMsgs == numTotalMsgs){
    // the parent will not send any more messages
    if(haveChildren){
      // tell children that they will not receive any more messages 
      for(int i = 0; i < NSUB; i++){
        pieces[childrenTreePieces[i]].recvTotalMsgCountsFromPieces(sentTo[i]);
      }
    }
    else{
      // don't have children, build own tree
      buildTree();
    }
    // FIXME - contribute to reduction. need callback 
    // maincb to main
    CkCallback cb(CkIndex_ParticleChunk::doneTreeBuild(), CkArrayIndex1D(0), chunks);
    contribute(0,0,CkReduction::concat,cb);
  }
}

void TreePiece::recvParticles(ParticleMsg *msg){ 
  bodyptr *particles = msg->particles;
  numRecvdMsgs++;

  if(myNumParticles+msg->num > MAX_PARTS_PER_TP && !haveChildren){
    // insert children into pieces array
    for(int i = 0; i < NSUB; i++){
      int child = NSUB*thisIndex+numTreePieces+i;
      pieces[child].insert((nodeptr)myRoot, i, false, myLevel >> 1);
      childrenTreePieces[i] = child;
    }
    haveChildren = true;
    myNumParticles = 0;
  }

  if(haveChildren){
    CkVec<CkVec<bodyptr> > partsToChild;
    partsToChild.resize(NSUB);

    int num = msg->num;
    int xp[NDIM];
    bodyptr p;
    for(int i = 0; i < num; i++){
      int c; // part i goes to child c
      p = particles[i]; 
      intcoord(xp,Pos(p));
      c = NSUB*thisIndex+numTreePieces+subindex(xp,Level(myRoot));
      partsToChild[c].push_back(particles[i]);
    }

    // at this point, we have a list of particles 
    // destined for each child
    for(int c = 0; c < NSUB; c++){
      int len = partsToChild[c].length();
      if(len > 0){
        // create msg from partsToChild[c], send
        ParticleMsg *msg = new (len) ParticleMsg;
        msg->num = len;
        memcpy(msg->particles, partsToChild[c].getVec(), len*sizeof(bodyptr));
        /*
        for(int i = 0; i < len; i++){
          msg->particles[i] = (partsToChild[c])[i]; 
        }
        */
        sentTo[c]++;
        pieces[childrenTreePieces[c]].recvParticles(msg);
      }
    }
  }
  else{
    myNumParticles += msg->num;
    // FIXME - add msg->particles to own particles
  }

  if(haveCounts){
    checkCompletion();
  }
  delete msg;
}

void TreePiece::buildTree(){
  bodyptr p, *pp;
  int ProcessId = thisIndex;
  nodeptr Current_Root;

  Current_Root = (nodeptr) myRoot;
  for(int i = 0; i < myParticles.length(); i++){
    Current_Root = (nodeptr) loadtree(myParticles[i], 
                                      (cellptr) Current_Root, 
                                      ProcessId);
  }
}

nodeptr TreePiece::loadtree(bodyptr p, cellptr root, unsigned int ProcessId){
  // FIXME - build local tree here myParticles
  int l, xq[NDIM], xp[NDIM], flag;
  int i, j, root_level;
  bool valid_root;
  int kidIndex;
  nodeptr *qptr, mynode;
  cellptr c;
  leafptr le;

  intcoord(xp, Pos(p));
  valid_root = TRUE;
  /*
  for (i = 0; i < NDIM; i++) {
    xor[i] = xp[i] ^ Local[ProcessId].Root_Coords[i];
  }
  for (i = IMAX >> 1; i > Level(root); i >>= 1) {
    for (j = 0; j < NDIM; j++) {
      if (xor[j] & i) {
        valid_root = FALSE;
        break;
      }
    }
    if (!valid_root) {
      break;
    }
  }
  if (!valid_root) {
    if (root != Global->G_root) {
      root_level = Level(root);
      for (j = i; j > root_level; j >>= 1) {
        root = (cellptr) Parent(root);
      }
      valid_root = TRUE;
      for (i = IMAX >> 1; i > Level(root); i >>= 1) {
        for (j = 0; j < NDIM; j++) {
          if (xor[j] & i) {
            valid_root = FALSE;
            break;
          }
        }
        if (!valid_root) {
          printf("P%d body %d\n", ProcessId, p - bodytab);
          root = Global->G_root;
        }
      }
    }
  }
  */
  mynode = (nodeptr) root;
  kidIndex = subindex(xp, Level(mynode));
  qptr = &Subp(mynode)[kidIndex];

  l = Level(mynode) >> 1;

  flag = TRUE;
  while (flag) {                           /* loop descending tree     */
    if (l == 0) {
      ckerr << "not enough levels in tree\n";
    }
    if (*qptr == NULL) { 
      /* lock the parent cell */
      //ALOCK(CellLock->CL, ((cellptr) mynode)->seqnum % MAXLOCK);
      //if (*qptr == NULL) {
        le = InitLeaf((cellptr) mynode, ProcessId);
        ParentOf(p) = (nodeptr) le;
        Level(p) = l;
        ChildNum(p) = le->num_bodies;
        ChildNum(le) = kidIndex;
        Bodyp(le)[le->num_bodies++] = p;
        *qptr = (nodeptr) le;
        flag = FALSE;
      //}
      //AULOCK(CellLock->CL, ((cellptr) mynode)->seqnum % MAXLOCK);
      /* unlock the parent cell */
    }
    if (flag && *qptr && (Type(*qptr) == LEAF)) {
      /*   reached a "leaf"?      */
      //ALOCK(CellLock->CL, ((cellptr) mynode)->seqnum % MAXLOCK);
      /* lock the parent cell */
      //if (Type(*qptr) == LEAF) {             /* still a "leaf"?      */
        le = (leafptr) *qptr;
        if (le->num_bodies == MAX_BODIES_PER_LEAF) {
          *qptr = (nodeptr) SubdivideLeaf(le, (cellptr) mynode, l,
              ProcessId);
        }
        else {
          ParentOf(p) = (nodeptr) le;
          Level(p) = l;
          ChildNum(p) = le->num_bodies;
          Bodyp(le)[le->num_bodies++] = p;
          flag = FALSE;
        }
      //}
      //AULOCK(CellLock->CL, ((cellptr) mynode)->seqnum % MAXLOCK);
      /* unlock the node           */
    }
    if (flag) {
      mynode = *qptr;
      kidIndex = subindex(xp, l);
      qptr = &Subp(*qptr)[kidIndex];  /* move down one level  */
      l = l >> 1;                            /* and test next bit    */
    }
  }
  //SETV(Root_Coords, xp);
  return ParentOf((leafptr) *qptr);

}

leafptr InitLeaf(cellptr parent, unsigned ProcessId)
//   cellptr parent;
//   unsigned ProcessId;
{
   leafptr l;
   int i, Mycell;

   l = makeleaf(ProcessId);
   //l->processor = ProcessId;
   l->next = NULL;
   l->prev = NULL;
   if (parent==NULL)
      Level(l) = IMAX >> 1;
   else
      Level(l) = Level(parent) >> 1;
   ParentOf(l) = (nodeptr) parent;
   ChildNum(l) = 0;
   return (l);
}

/*
 * MAKECELL: allocation routine for cells.
 */

cellptr makecell(unsigned ProcessId)
{
   cellptr c;
   int i, Mycell;
    
   /*
   if (mynumcell == maxmycell) {
      error("makecell: Proc %d needs more than %d cells; increase fcells\n", 
	    ProcessId,maxmycell);
   }
   Mycell = mynumcell++;
   c = ctab + Mycell;
   c->seqnum = ProcessId*maxmycell+Mycell;
   */
   c = new cell;
   Type(c) = CELL;
   Done(c) = FALSE;
   Mass(c) = 0.0;
   for (i = 0; i < NSUB; i++) {
      Subp(c)[i] = NULL;
   }
   //mycelltab[myncell++] = c;
   return (c);
}

/*
 * MAKELEAF: allocation routine for leaves.
 */

leafptr makeleaf(unsigned ProcessId)
{
   leafptr le;
   int i, Myleaf;
    
    /*
   if (mynumleaf == maxmyleaf) {
      error("makeleaf: Proc %d needs more than %d leaves; increase fleaves\n",
	    ProcessId,maxmyleaf);
   }
   Myleaf = mynumleaf++;
   le = ltab + Myleaf;
   le->seqnum = ProcessId * maxmyleaf + Myleaf;
   */
   Type(le) = LEAF;
   Done(le) = FALSE;
   Mass(le) = 0.0;
   le->num_bodies = 0;
   for (i = 0; i < MAX_BODIES_PER_LEAF; i++) {
      Bodyp(le)[i] = NULL;
   }
   //myleaftab[mynleaf++] = le;
   return (le);
}

cellptr
TreePiece::SubdivideLeaf (leafptr le, cellptr parent, unsigned int l, unsigned int ProcessId)
/*
   leafptr le;
   cellptr parent;
   unsigned int l;
   unsigned int ProcessId;
*/
{
   cellptr c;
   int i, index;
   int xp[NDIM];
   bodyptr bodies[MAX_BODIES_PER_LEAF];
   int num_bodies;
   bodyptr p;

   /* first copy leaf's bodies to temp array, so we can reuse the leaf */
   num_bodies = le->num_bodies;
   for (i = 0; i < num_bodies; i++) {
      bodies[i] = Bodyp(le)[i];
      Bodyp(le)[i] = NULL;
   }
   le->num_bodies = 0;
   /* create the parent cell for this subtree */
   c = InitCell(parent, ProcessId);
   ChildNum(c) = ChildNum(le);
   /* do first particle separately, so we can reuse le */
   p = bodies[0];
   intcoord(xp, Pos(p));
   index = subindex(xp, l);
   Subp(c)[index] = (nodeptr) le;
   ChildNum(le) = index;
   ParentOf(le) = (nodeptr) c;
   Level(le) = l >> 1;
   /* set stuff for body */
   ParentOf(p) = (nodeptr) le;
   ChildNum(p) = le->num_bodies;
   Level(p) = l >> 1;
   /* insert the body */
   Bodyp(le)[le->num_bodies++] = p;
   /* now handle the rest */
   for (i = 1; i < num_bodies; i++) {
      p = bodies[i];
      intcoord(xp, Pos(p));
      index = subindex(xp, l);
      if (!Subp(c)[index]) {
	 le = InitLeaf(c, ProcessId);
	 ChildNum(le) = index;
	 Subp(c)[index] = (nodeptr) le;
      }
      else {
	 le = (leafptr) Subp(c)[index];
      }
      ParentOf(p) = (nodeptr) le;
      ChildNum(p) = le->num_bodies;
      Level(p) = l >> 1;
      Bodyp(le)[le->num_bodies++] = p;
   }
   return c;
}

cellptr InitCell(cellptr parent, unsigned ProcessId)
/*
   cellptr parent;
      unsigned ProcessId;
*/
{
  cellptr c;
  int i, Mycell;

  c = makecell(ProcessId);
  //c->processor = ProcessId;
  c->next = NULL;
  c->prev = NULL;
  if (parent == NULL)
    Level(c) = IMAX >> 1;
  else
    Level(c) = Level(parent) >> 1;
  ParentOf(c) = (nodeptr) parent;
  ChildNum(c) = 0;
  return (c);
}

/*
 * SUBINDEX: determine which subcell to select.
 */

int subindex(int x[NDIM], int l)
{
   int i, k;
   int yes;
    
   i = 0;
   yes = FALSE;
   if (x[0] & l) {
      i += NSUB >> 1;
      yes = TRUE;
   }
   for (k = 1; k < NDIM; k++) {
      if (((x[k] & l) && !yes) || (!(x[k] & l) && yes)) { 
	 i += NSUB >> (k + 1);
	 yes = TRUE;
      }
      else yes = FALSE;
   }

   return (i);
}

/* * INTCOORD: compute integerized coordinates.  * Returns: TRUE
unless rp was out of bounds.  */

bool intcoord(int *xp, vector rp)
{
   int k;
   bool inb;
   double xsc;
   double tmp;
    
   inb = TRUE;
   for (k = 0; k < NDIM; k++) {
      xsc = (rp[k] - rmin[k]) / rsize; 
      if (0.0 <= xsc && xsc < 1.0) {
        tmp = IMAX * xsc;
	 xp[k] = (int)tmp;
      }
      else {
	 inb = FALSE;
      }
   }
   return (inb);
}

