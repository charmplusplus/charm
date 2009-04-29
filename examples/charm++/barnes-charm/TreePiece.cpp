#include "barnes.h"

TreePiece::TreePiece(CmiUInt8 p_, int which, bool isTopLevel_, int level_, CkArrayIndex1D parentIdx){
  isTopLevel = isTopLevel_;
  nodeptr p;
  myLevel = level_;
  if(!isTopLevel_){
    whichChildAmI = which;
    parentIndex = parentIdx;
    // save parent
    parent = (nodeptr)p_;
  }
  else{
    // don't save parent if top-level tp. parent will be set by
    // acceptroots
    whichChildAmI = thisIndex;
  }

  mycelltab.reserve(maxmycell);
  myleaftab.reserve(maxmyleaf);

  myRoot = NULL;

  numTotalMsgs = -1;
  numRecvdMsgs = 0;
  haveChildren = false;
  myNumParticles = 0;
  
  myncell = 0;
  mynleaf = 0;

  haveCounts = false;

  for(int i = 0; i < NSUB; i++){
    sentTo[i] = 0;
  }

#ifdef MEMCHECK
  CkPrintf("piece %d after construction\n", thisIndex);
  CmiMemoryCheck();
#endif
}

void TreePiece::acceptRoots(CmiUInt8 roots_, CkCallback &cb){
  if(isTopLevel){
#ifdef MEMCHECK
  CkPrintf("piece %d before acceptRoots \n", thisIndex);
  CmiMemoryCheck();
#endif
    nodeptr *pp = (nodeptr *)roots_;
    nodeptr p = pp[thisIndex/NSUB];
    parent = p;
#ifdef MEMCHECK
  CkPrintf("piece %d after acceptRoots\n", thisIndex);
  CmiMemoryCheck();
#endif

    CkPrintf("piece [%d] acceptRoot parent 0x%x\n", thisIndex, parent);
  }
  contribute(0,0,CkReduction::concat,cb);
}

void TreePiece::recvTotalMsgCountsFromPieces(int totalNumFromParent){
#ifdef MEMCHECK
  CkPrintf("piece %d before recvTotalMsgCountsFromPieces\n", thisIndex);
  CmiMemoryCheck();
#endif
  CkAssert(!isTopLevel);
  numTotalMsgs = totalNumFromParent;
  CkPrintf("piece %d got count from PARENT: %d\n", thisIndex, numTotalMsgs);
  haveCounts = true;
  checkCompletion();
#ifdef MEMCHECK
  CkPrintf("piece %d after recvTotalMsgCountsFromPieces\n", thisIndex);
  CmiMemoryCheck();
#endif
}


void TreePiece::recvTotalMsgCountsFromChunks(CkReductionMsg *msg){
  if(isTopLevel){
#ifdef MEMCHECK
    CkPrintf("piece %d before recvTotalMsgCountsFromChunks\n", thisIndex);
    CmiMemoryCheck();
#endif
    int *data = (int *)msg->getData();
    int nelts = msg->getSize()/sizeof(int);

    numTotalMsgs = data[thisIndex]; /* between 0 and numTreePieces, 
                                       which is the number of 
                                       top-level treepieces */
    CkPrintf("piece %d got count from CHUNKS: %d (%d)\n", thisIndex, numTotalMsgs, nelts);
    haveCounts = true;
    checkCompletion();
#ifdef MEMCHECK
    CkPrintf("piece %d after recvTotalMsgCountsFromChunks\n", thisIndex);
    CmiMemoryCheck();
#endif
  }
  delete msg;
}

void TreePiece::checkCompletion(){
#ifdef MEMCHECK
  CkPrintf("piece %d before checkCompletion \n", thisIndex);
  CmiMemoryCheck();
#endif
  CkPrintf("piece %d checkcompletion, recvd %d, total %d\n", thisIndex, numRecvdMsgs, numTotalMsgs);
  if(numRecvdMsgs == numTotalMsgs){
    CkPrintf("piece %d has all particles\n", thisIndex);
    // the parent will not send any more messages
    if(haveChildren){
      // tell children that they will not receive any more messages 
      for(int i = 0; i < NSUB; i++){
        int child = childrenTreePieces[i];
        CkPrintf("piece %d -> child %d, total messages sentTo: %d\n", thisIndex, child, sentTo[i]);
        pieces[child].recvTotalMsgCountsFromPieces(sentTo[i]);
      }
      CkPrintf("piece %d fake doneTreeBuild()\n", thisIndex);
    }
    else{
      // don't have children, build own tree
      CkPrintf("piece %d doesn't have children, building tree\n", thisIndex);
      buildTree();
      hackcofm(0,thisIndex);
      if(!isTopLevel){
        // once you've built your own tree, 
        // you must notify your parent that you're done
        CkPrintf("piece %d real !topLevel doneTreeBuild()\n", thisIndex);
        pieces[parentIndex].childDone(whichChildAmI);
      }
      else{
        CkPrintf("piece %d real topLevel doneTreeBuild()\n", thisIndex);
      }
      CkCallback cb(CkIndex_ParticleChunk::doneTreeBuild(), CkArrayIndex1D(0), chunks);
      contribute(0,0,CkReduction::concat,cb);
    }
  }
#ifdef MEMCHECK
  CkPrintf("piece %d after checkCompletion\n", thisIndex);
  CmiMemoryCheck();
#endif
}

void TreePiece::childDone(int which){
  pendingChildren--;
  
  // 'which' child just finished building its tree (and hence calculating
  // moments. add these to your own. 
  CkPrintf("piece %d child %d done, updateMoments\n", thisIndex, which);
  updateMoments(which);

  if(pendingChildren == 0){
    DIVVS(Pos(myRoot), Pos(myRoot), Mass(myRoot));
#ifdef MEMCHECK
    CkPrintf("piece %d before childDone \n", thisIndex);
    CmiMemoryCheck();
#endif
    CkPrintf("piece %d all children done\n", thisIndex);
    if(!isTopLevel){
      // talk to parent
      CkPrintf("piece %d (whichChildAmI: %d) -> parent %d, i'm done\n", thisIndex, whichChildAmI, parentIndex.index);
      pieces[parentIndex].childDone(whichChildAmI);
    }
    CkCallback cb(CkIndex_ParticleChunk::doneTreeBuild(), CkArrayIndex1D(0), chunks);
    contribute(0,0,CkReduction::concat,cb);
#ifdef MEMCHECK
    CkPrintf("piece %d after childDone\n", thisIndex);
    CmiMemoryCheck();
#endif
  }
}

void TreePiece::recvParticles(ParticleMsg *msg){ 
  bodyptr *particles = msg->particles;
  numRecvdMsgs++;

#ifdef MEMCHECK
  CkPrintf("piece %d before recvParticles \n", thisIndex);
  CmiMemoryCheck();
#endif
  CkPrintf("piece %d recvd %d particles, numRecvdMsgs: %d\n", thisIndex, msg->num, numRecvdMsgs);
  /*
  for(int i = 0; i < msg->num; i++){
    CkPrintf("piece %d 0x%x\n", thisIndex, msg->particles[i]);
  }
  */

  int newtotal = myNumParticles+msg->num;

  if(newtotal > maxPartsPerTp && newtotal > MAX_BODIES_PER_LEAF && !haveChildren){
    // insert children into pieces array
    CkPrintf("piece %d has too many (%d+%d) particles; creating children\n", thisIndex, myNumParticles, msg->num);
    // first create own root
    myRoot = (nodeptr) InitCell((cellptr)parent, thisIndex);
    Subp(parent)[whichChildAmI] = myRoot;
    Mass(myRoot) = 0.0;
    Cost(myRoot) = 0;
    CLRV(Pos(myRoot));

    for(int i = 0; i < NSUB; i++){
      int child = NSUB*thisIndex+numTreePieces+i;
      CkPrintf("piece %d inserting child %d\n", thisIndex, child);
      pieces[child].insert((CmiUInt8)myRoot, i, false, myLevel >> 1, thisIndex);
      childrenTreePieces[i] = child;
    }
    haveChildren = true;
    pendingChildren = NSUB;
    /*
    myNumParticles = 0;
    myParticles.free();
    */
  }

  if(haveChildren){
    CkPrintf("piece %d has children\n", thisIndex);
    CkVec<CkVec<bodyptr> > partsToChild;
    partsToChild.resize(NSUB);
    //partsToChild.length() = 0;

    int num = msg->num;
    int xp[NDIM];
    bodyptr p;
    // first get particles that i already have
    for(int i = 0; i < myNumParticles; i++){
      int c; // part i goes to child c
      int relc; // index of child relative to this node (0..NSUB)
      p = myParticles[i]; 
      CkAssert(intcoord(xp,Pos(p)));
      relc = subindex(xp,Level(myRoot));
      //c = NSUB*thisIndex + numTreePieces + relc;
      partsToChild[relc].push_back(myParticles[i]);
    }
    myNumParticles = 0;
    myParticles.free();

    // then get contents of this message
    for(int i = 0; i < num; i++){
      int c; // part i goes to child c
      int relc; // index of child relative to this node (0..NSUB)
      p = particles[i]; 
      CkAssert(intcoord(xp,Pos(p)));
      relc = subindex(xp,Level(myRoot));
      //c = NSUB*thisIndex + numTreePieces + relc;
      partsToChild[relc].push_back(particles[i]);
    }

    // at this point, we have a list of particles 
    // destined for each child
    for(int c = 0; c < NSUB; c++){
      int len = partsToChild[c].length();
      if(len > 0){
        // create msg from partsToChild[c], send
        ParticleMsg *amsg = new (len) ParticleMsg;
        amsg->num = len;
        memcpy(amsg->particles, partsToChild[c].getVec(), len*sizeof(bodyptr));
        /*
           for(int i = 0; i < len; i++){
           msg->particles[i] = (partsToChild[c])[i]; 
           }
           */
        sentTo[c]++;
        int tochild = childrenTreePieces[c];
        pieces[tochild].recvParticles(amsg);
        CkPrintf("piece %d sent %d particles to child %d. sentTo[%d] = %d\n", thisIndex, len, tochild, c, sentTo[c]);

        /*
        for(int i = 0; i < amsg->num; i++){
          CkPrintf("piece %d 0x%x\n", thisIndex, amsg->particles[i]);
        }
        */
      }
    }
  }
  else{
    CkPrintf("piece %d adding %d particles to self (%d), total: %d\n", thisIndex, msg->num, myNumParticles, myNumParticles+msg->num);
    
    // this is how many particles we had before expanding
    int savedpos = myNumParticles;
    // we now have msg->num additional particles
    myNumParticles += msg->num;
    // expand array of particles to include new ones
    myParticles.resize(myNumParticles);
    // this is where we start copying new particles to 
    bodyptr *savedstart = myParticles.getVec()+savedpos;

    memcpy(savedstart, msg->particles, (msg->num)*sizeof(bodyptr));

    /*
    for(int i = 0; i < msg->num; i++){
      myParticles.push_back(msg->particles[i]);
    }
    */
  }

  if(haveCounts){
    checkCompletion();
  }
  delete msg;
#ifdef MEMCHECK
  CkPrintf("piece %d after recvParticles\n", thisIndex);
  CmiMemoryCheck();
#endif
}

void TreePiece::buildTree(){
  bodyptr p, *pp;
  int ProcessId = thisIndex;
  nodeptr Current_Root;

#ifdef MEMCHECK
  CkPrintf("piece %d before buildTree\n", thisIndex);
  CmiMemoryCheck();
#endif
  // start from parent 
  // this way, particles can be added to Root until it
  // needs to be split.
  Current_Root = (nodeptr) parent;
  for(int i = 0; i < myParticles.length(); i++){
    /*
    if(thisIndex == 5){
      CkPrintf("piece %d inserting particle %d level: %d\n", thisIndex, i, Level(Current_Root));
    }
    */
    Current_Root = (nodeptr) loadtree(myParticles[i], (cellptr) Current_Root, ProcessId);
    /*
    if(firstInsert || justSplit){
      yRoot = Current_Root;
    }
    */
  }

  /*
  nodeptr pr;
  if(myNumParticles > 0){
    while(Current_Root != parent){
      pr = Current_Root;
      Current_Root = ParentOf(Current_Root);
    }
    myRoot = pr;
  }
  else{
    myRoot = NULL;
  }
  CkPrintf("piece %d myRoot: 0x%x, parent: 0x%x\n", thisIndex, myRoot, parent);
  */
#ifdef MEMCHECK
  CkPrintf("piece %d after buildTree\n", thisIndex);
  CmiMemoryCheck();
#endif
}

nodeptr TreePiece::loadtree(bodyptr p, cellptr root, unsigned int ProcessId){
  int l, xq[NDIM], xp[NDIM], flag;
  int i, j, root_level;
  bool valid_root;
  int kidIndex;
  nodeptr *qptr, mynode;
  cellptr c;
  leafptr le;

#ifdef MEMCHECK
  CkPrintf("piece %d before loadtree\n", thisIndex);
  CmiMemoryCheck();
#endif
  CkAssert(intcoord(xp, Pos(p)));
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
      CkAbort("not enough levels in tree\n");
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

        // this was the first time a leaf was given a particle.
        // this must be the first leaf, and should be saved as
        // yRoot.
        /*
        if(!firstInsert){
          firstInsert = true;
        }
        */
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
          *qptr = (nodeptr) SubdivideLeaf(le, (cellptr) mynode, l, ProcessId);
          // the first time a leaf is split and a parent
          // cell is created. this should be saved as yRoot
          /*
          if(!justSplit){
            justSplit = true;
          }
          */
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
#ifdef MEMCHECK
  CkPrintf("piece %d after loadtree\n", thisIndex);
  CmiMemoryCheck();
#endif
  return ParentOf((leafptr) *qptr);

}

leafptr TreePiece::InitLeaf(cellptr parent, unsigned ProcessId)
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

    /*
   if (mynleaf == maxmyleaf) {
      CkPrintf("makeleaf: piece %d needs more than %d leaves; increase fleaves\n",ProcessId,maxmyleaf);
      CkAbort("fleaves\n");
   }
   */
   
   myleaftab.push_back(l);
   mynleaf++;

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
   le = new leaf; 
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
TreePiece::SubdivideLeaf (leafptr le, cellptr parent_, unsigned int l, unsigned int ProcessId)
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
   c = InitCell(parent_, ProcessId);
   ChildNum(c) = ChildNum(le);
   /* do first particle separately, so we can reuse le */
   p = bodies[0];
   CkAssert(intcoord(xp, Pos(p)));
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
      CkAssert(intcoord(xp, Pos(p)));
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

cellptr TreePiece::InitCell(cellptr parent, unsigned ProcessId)
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
   
   /*
   if (myncell == maxmycell) {
      CkPrintf("makecell: piece %d needs more than %d cells; increase fcells\n", ProcessId,maxmycell);
   }
   */
   
   mycelltab.push_back(c);
   myncell++;

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

/*
 * HACKCOFM: descend tree finding center-of-mass coordinates.
 */

void TreePiece::hackcofm(int nc, unsigned ProcessId)
{
   int i,Myindex;
   nodeptr r;
   leafptr l;
   leafptr* ll;
   bodyptr p;
   cellptr q;
   cellptr *cc;
   vector tmpv, dr;
   real drsq;
   matrix drdr, Idrsq, tmpm;

   /* get a cell using get*sub.  Cells are got in reverse of the order in */
   /* the cell array; i.e. reverse of the order in which they were created */
   /* this way, we look at child cells before parents			 */
    
   for (ll = myleaftab.getVec() + mynleaf - 1; ll >= myleaftab.getVec(); ll--) {
      l = *ll;
      Mass(l) = 0.0;
      Cost(l) = 0;
      CLRV(Pos(l));
      for (i = 0; i < l->num_bodies; i++) {
	 p = Bodyp(l)[i];
	 Mass(l) += Mass(p);
	 Cost(l) += Cost(p);
	 MULVS(tmpv, Pos(p), Mass(p));
	 ADDV(Pos(l), Pos(l), tmpv);
      }
      DIVVS(Pos(l), Pos(l), Mass(l));
#ifdef QUADPOLE
      CLRM(Quad(l));
      for (i = 0; i < l->num_bodies; i++) {
	 p = Bodyp(l)[i];
	 SUBV(dr, Pos(p), Pos(l));
	 OUTVP(drdr, dr, dr);
	 DOTVP(drsq, dr, dr);
	 SETMI(Idrsq);
	 MULMS(Idrsq, Idrsq, drsq);
	 MULMS(tmpm, drdr, 3.0);
	 SUBM(tmpm, tmpm, Idrsq);
	 MULMS(tmpm, tmpm, Mass(p));
	 ADDM(Quad(l), Quad(l), tmpm);
      }
#endif
      Done(l)=TRUE;
   }
   for (cc = mycelltab.getVec()+myncell-1; cc >= mycelltab.getVec(); cc--) {
      q = *cc;
      Mass(q) = 0.0;
      Cost(q) = 0;
      CLRV(Pos(q));
      for (i = 0; i < NSUB; i++) {
	 r = Subp(q)[i];
	 if (r != NULL) {
           /*
	    while(!Done(r)) {
	       // wait
	    }
            */
	    Mass(q) += Mass(r);
	    Cost(q) += Cost(r);
	    MULVS(tmpv, Pos(r), Mass(r));
	    ADDV(Pos(q), Pos(q), tmpv);
	    Done(r) = FALSE;
	 }
      }
      DIVVS(Pos(q), Pos(q), Mass(q));
#ifdef QUADPOLE
      CLRM(Quad(q));
      for (i = 0; i < NSUB; i++) {
	 r = Subp(q)[i];
	 if (r != NULL) {
	    SUBV(dr, Pos(r), Pos(q));
	    OUTVP(drdr, dr, dr);
	    DOTVP(drsq, dr, dr);
	    SETMI(Idrsq);
	    MULMS(Idrsq, Idrsq, drsq);
	    MULMS(tmpm, drdr, 3.0);
	    SUBM(tmpm, tmpm, Idrsq);
	    MULMS(tmpm, tmpm, Mass(r));
	    ADDM(tmpm, tmpm, Quad(r));
	    ADDM(Quad(q), Quad(q), tmpm);
	 }
      }
#endif
      //Done(q)=TRUE;
   }
}

void TreePiece::updateMoments(int which){

  int i,Myindex;
  nodeptr r;
  leafptr l;
  leafptr* ll;
  bodyptr p;
  cellptr q;
  cellptr *cc;
  vector tmpv, dr;
  real drsq;
  matrix drdr, Idrsq, tmpm;

  // add moments of child 'which' to meRoot
  q = (cellptr) myRoot;
  r = Subp(q)[which];
  if (r != NULL){
    Mass(q) += Mass(r);
    Cost(q) += Cost(r);
    MULVS(tmpv, Pos(r), Mass(r));
    ADDV(Pos(q), Pos(q), tmpv);
    //Done(r) = FALSE;

#ifdef QUADPOLE
    CLRM(Quad(q));
    r = Subp(q)[which];
    CkAssert (r != NULL);
    SUBV(dr, Pos(r), Pos(q));
    OUTVP(drdr, dr, dr);
    DOTVP(drsq, dr, dr);
    SETMI(Idrsq);
    MULMS(Idrsq, Idrsq, drsq);
    MULMS(tmpm, drdr, 3.0);
    SUBM(tmpm, tmpm, Idrsq);
    MULMS(tmpm, tmpm, Mass(r));
    ADDM(tmpm, tmpm, Quad(r));
    ADDM(Quad(q), Quad(q), tmpm);
#endif

    //Done(q)=TRUE;
  }
}

