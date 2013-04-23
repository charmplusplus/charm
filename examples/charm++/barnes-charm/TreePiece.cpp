#include "barnes.h"

TreePiece::TreePiece(CmiUInt8 p_, int which, int level_,  real rx, real ry, real rz, real rs, CkArrayIndex1D parentIdx){
  nodeptr p;
  isTopLevel = false;

  usesAtSync = false;
  setMigratable(false);
  
  rmin[0] = rx;
  rmin[1] = ry;
  rmin[2] = rz;

  rsize = rs;

  myLevel = level_;
  whichChildAmI = which;
  parentIndex = parentIdx;
  // save parent
  parent = (nodeptr)p_;
  haveParent = true;

  mycelltab.reserve(maxmycell);
  myleaftab.reserve(maxmyleaf);
  ctab = new cell [maxmycell];
  ltab = new leaf [maxmyleaf];

  myRoot = NULL;

  numTotalMsgs = -1;
  numRecvdMsgs = 0;
  haveChildren = false;
  myNumParticles = 0;

  myncell = 0;
  mynleaf = 0;

  haveCounts = false;
  pendingChildren = 0;
  partsToChild.resize(NSUB);

  for(int i = 0; i < NSUB; i++){
    sentTo[i] = 0;
    childrenTreePieces[i] = -1;
    isPending[i] = false;
    partsToChild[i].reserve(INIT_PARTS_PER_CHILD);
  }
  
  wantToSplit = false;

#ifdef MEMCHECK
  CkPrintf("piece %d after construction\n", thisIndex);
  CmiMemoryCheck();
#endif
}

TreePiece::TreePiece(CmiUInt8 p_, int which, int level_, CkArrayIndex1D parentIdx){
  isTopLevel = true;
  nodeptr p;
  myLevel = level_;
  // don't save parent if top-level tp. parent will be set by
  // acceptroots
  whichChildAmI = thisIndex%NSUB;

  usesAtSync = false;
  setMigratable(false);

  mycelltab.reserve(maxmycell);
  myleaftab.reserve(maxmyleaf);
  ctab = new cell [maxmycell];
  ltab = new leaf [maxmyleaf];

  myRoot = NULL;

  numTotalMsgs = -1;
  numRecvdMsgs = 0;
  haveChildren = false;
  myNumParticles = 0;

  myncell = 0;
  mynleaf = 0;

  haveCounts = false;
  pendingChildren = 0;
  partsToChild.resize(NSUB);

  for(int i = 0; i < NSUB; i++){
    sentTo[i] = 0;
    childrenTreePieces[i] = -1;
    isPending[i] = false;
    partsToChild[i].reserve(INIT_PARTS_PER_CHILD);
  }
  
  wantToSplit = false;

#ifdef MEMCHECK
  CkPrintf("piece %d after construction\n", thisIndex);
  CmiMemoryCheck();
#endif
}

void TreePiece::processParticles(bodyptr *particles, int num){
  // get contents of this message
  int xp[NDIM];
  bodyptr p;

  for(int i = 0; i < num; i++){
    int c; // part i goes to child c
    int relc; // index of child relative to this node (0..NSUB)
    p = particles[i]; 
    intcoord(xp,Pos(p),rmin,rsize);
    relc = subindex(xp,Level(myRoot));
    partsToChild[relc].push_back(particles[i]);
  }
}

void TreePiece::sendParticlesToChildren(){
  for(int i = 0; i < NSUB; i++){
    int len = partsToChild[i].length();
    if(childrenTreePieces[i] < 0 && len > 0){
      CkAssert(!isPending[i]);
      int child = NSUB*thisIndex+numTreePieces+i;
      // TODO
      // common code outside
      // chares or chare arrays?
      // remote method instead of message sends
      pieces[child].insert((CmiUInt8)myRoot, i, myLevel >> 1, rmin[0], rmin[1], rmin[2], rsize, thisIndex);
      childrenTreePieces[i] = child;
      pendingChildren++;
      isPending[i] = true;
#ifdef VERBOSE_PIECES
      CkPrintf("piece %d inserting child %d, pending children: %d\n", thisIndex, child, pendingChildren);
#endif
      // create msg from partsToChild[i], send
      ParticleMsg *amsg = new (len) ParticleMsg;
      amsg->num = len;
      memcpy(amsg->particles, partsToChild[i].getVec(), len*sizeof(bodyptr));
      pieces[child].recvParticles(amsg);
#ifdef VERBOSE_PIECES
      CkPrintf("piece %d sent %d particles to child %d. sentTo[%d] = %d\n", thisIndex, len, child, i, sentTo[i]);
#endif
      partsToChild[i].length() = 0;
    }
    else if(len > 0){
      int child = childrenTreePieces[i];
      if(!isPending[i]){
        pendingChildren++;
#ifdef VERBOSE_PIECES
        CkPrintf("piece %d has child %d, pending children: %d\n", thisIndex, child, pendingChildren);
#endif
        isPending[i] = true;
      }
      pieces[child].recvRootFromParent((CmiUInt8)myRoot, rmin[0], rmin[1], rmin[2], rsize);
      // create msg from partsToChild[i], send
      ParticleMsg *amsg = new (len) ParticleMsg;
      amsg->num = len;
      memcpy(amsg->particles, partsToChild[i].getVec(), len*sizeof(bodyptr));
      pieces[child].recvParticles(amsg);
#ifdef VERBOSE_PIECES
      CkPrintf("piece %d sent %d particles to child %d. sentTo[%d] = %d\n", thisIndex, len, child, i, sentTo[i]);
#endif
      partsToChild[i].length() = 0;
    }
  }
}

void TreePiece::resetPartsToChild(){
  for(int i = 0; i < NSUB; i++){
    partsToChild[i].length() = 0;
  }
}

void TreePiece::recvRootFromParent(CmiUInt8 r, real rx, real ry, real rz, real rs){
  parent = (nodeptr)r;
  haveParent = true;
  rmin[0] = rx;
  rmin[1] = ry;
  rmin[2] = rz;
  rsize = rs;

#ifdef VERBOSE_PIECES
      CkPrintf("piece %d recvd root from parent\n", thisIndex);
#endif

  if(wantToSplit){
#ifdef VERBOSE_PIECES
      CkPrintf("piece %d wanted to split\n", thisIndex);
#endif
    // create own root
    myRoot = (nodeptr) InitCell((cellptr)parent, thisIndex);
    Subp(parent)[whichChildAmI] = myRoot;
    Mass(myRoot) = 0.0;
    Cost(myRoot) = 0;
    CLRV(Pos(myRoot));
    NodeKey(myRoot) = (NodeKey(parent) << NDIM)+whichChildAmI;

    // process own particles
    processParticles(myParticles.getVec(), myNumParticles);
    myNumParticles = 0;
    myParticles.length() = 0; 

    // process messages
    for(int i = 0; i < bufferedMsgs.length(); i++){
      processParticles(bufferedMsgs[i]->particles, bufferedMsgs[i]->num);
      delete bufferedMsgs[i];
    }

    // send out all particles
    sendParticlesToChildren();
  }
#ifdef MEMCHECK
  CkPrintf("piece %d after recvParticles (recvRootFromParent)\n", thisIndex);
  CmiMemoryCheck();
#endif
}

void TreePiece::acceptRoots(CmiUInt8 roots_, real rsize_, real rmx, real rmy, real rmz, CkCallback &cb){
  rsize = rsize_;
  rmin[0] = rmx;
  rmin[1] = rmy;
  rmin[2] = rmz;

#ifdef VERBOSE_PIECES
  CkPrintf("piece %d acceptRoot rmin: (%f,%f,%f) rsize: %f\n", thisIndex, rmin[0], rmin[1], rmin[2], rsize);
#endif

  if(isTopLevel){
#ifdef MEMCHECK
  CkPrintf("piece %d before acceptRoots \n", thisIndex);
  CmiMemoryCheck();
#endif
    nodeptr *pp = (nodeptr *)roots_;
    nodeptr p = pp[thisIndex/NSUB];
    parent = p;
    haveParent = true;
#ifdef MEMCHECK
  CkPrintf("piece %d after acceptRoots\n", thisIndex);
  CmiMemoryCheck();
#endif

#ifdef VERBOSE_PIECES
    CkPrintf("piece [%d] acceptRoot parent 0x%x (%ld)\n", thisIndex, parent, NodeKey(parent));
#endif
  }
  contribute(0,0,CkReduction::concat,cb);
}

void TreePiece::childDone(int which){
  pendingChildren--;
  
  // 'which' child just finished building its tree (and hence calculating
  // moments. add these to your own. 
#ifdef VERBOSE_PIECES
  CkPrintf("piece %d child %d done, updateMoments, pendingChildren: %d\n", thisIndex, which, pendingChildren);
#endif
  updateMoments(which);

  if(pendingChildren == 0){
    DIVVS(Pos(myRoot), Pos(myRoot), Mass(myRoot));
#ifdef MEMCHECK
    CkPrintf("piece %d before childDone \n", thisIndex);
    CmiMemoryCheck();
#endif
#ifdef VERBOSE_PIECES
    CkPrintf("piece %d all children done\n", thisIndex);
#endif
    if(!isTopLevel){
      // talk to parent
#ifdef VERBOSE_PIECES
      CkPrintf("piece %d (whichChildAmI: %d) -> parent %d, i'm done\n", thisIndex, whichChildAmI, parentIndex.index);
#endif
      pieces[parentIndex].childDone(whichChildAmI);
    }
    CkCallback cb(CkIndex_ParticleChunk::doneTreeBuild(), CkArrayIndex1D(0), chunks);
    contribute(0,0,CkReduction::concat,cb);
#ifdef MEMCHECK
    CkPrintf("piece %d after childDone\n", thisIndex);
    CmiMemoryCheck();
#endif
  }
  else if(pendingChildren < 0){
    CkAbort("pendingChildren < 0\n");
  }
}

void TreePiece::recvParticles(ParticleMsg *msg){ 
  bodyptr *particles = msg->particles;

#ifdef MEMCHECK
  CkPrintf("piece %d before recvParticles \n", thisIndex);
  CmiMemoryCheck();
#endif

  int newtotal = myNumParticles+msg->num;

  if(newtotal > maxPartsPerTp && !haveChildren){
    // insert children into pieces array
#ifdef VERBOSE_PIECES
    CkPrintf("piece %d has too many (%d+%d) particles; creating children\n", thisIndex, myNumParticles, msg->num);
#endif
    if(haveParent){
#ifdef VERBOSE_PIECES
      CkPrintf("piece %d has parent, process\n", thisIndex);
#endif
      // first create own root
      // then process message and own particles, to see where they should go
      myRoot = (nodeptr) InitCell((cellptr)parent, thisIndex);
      Subp(parent)[whichChildAmI] = myRoot;
      Mass(myRoot) = 0.0;
      Cost(myRoot) = 0;
      CLRV(Pos(myRoot));
      NodeKey(myRoot) = (NodeKey(parent) << NDIM)+whichChildAmI;

      // process own particles
      processParticles(myParticles.getVec(), myNumParticles);
      myNumParticles = 0;
      myParticles.length() = 0; 

      // process message 
      processParticles(msg->particles, msg->num);
      haveChildren = true;

      // send out all particles
      sendParticlesToChildren();
    }
    else{
#ifdef VERBOSE_PIECES
      CkPrintf("piece %d doesn't have parent, buffer msg\n", thisIndex);
#endif
      // buffer message and return
      bufferedMsgs.push_back(msg);
      wantToSplit = true;
      return;
    }
  }
  else if(!haveChildren){
    // stuff msg into own parts
#ifdef VERBOSE_PIECES
    CkPrintf("piece %d adding %d particles to self (%d), total: %d\n", thisIndex, msg->num, myNumParticles, myNumParticles+msg->num);
#endif

    // this is how many particles we had before expanding
    int savedpos = myNumParticles;
    // we now have msg->num additional particles
    myNumParticles += msg->num;
    // expand array of particles to include new ones
    myParticles.resize(myNumParticles);
    // this is where we start copying new particles to 
    bodyptr *savedstart = myParticles.getVec()+savedpos;

    memcpy(savedstart, msg->particles, (msg->num)*sizeof(bodyptr));
  }
  else{
    // have no particles of own, send to children
#ifdef VERBOSE_PIECES
    CkPrintf("piece %d has children\n", thisIndex);
#endif
    // at this point, we have a list of particles 
    // destined for each child
    processParticles(msg->particles, msg->num);
    sendParticlesToChildren();
    
  }

  delete msg;
#ifdef MEMCHECK
  CkPrintf("piece %d after recvParticles\n", thisIndex);
  CmiMemoryCheck();
#endif
}

void TreePiece::doBuildTree(){
  if(haveChildren){
#ifdef VERBOSE_PIECES
    CkPrintf("piece %d fake doneTreeBuild()\n", thisIndex);
#endif
  }
  else{
    // don't have children, build own tree
#ifdef VERBOSE_PIECES
    CkPrintf("piece %d doesn't have children, building tree\n", thisIndex);
#endif
    buildTree();
    hackcofm(0,thisIndex);
    if(!isTopLevel && haveParent){
      // once you've built your own tree, 
      // you must notify your parent that you're done
#ifdef VERBOSE_PIECES
      CkPrintf("piece %d real !topLevel doneTreeBuild(), haveParent: %d\n", thisIndex, haveParent);
#endif
      pieces[parentIndex].childDone(whichChildAmI);
    }
#ifdef VERBOSE_PIECES
    else{
      CkPrintf("piece %d real topLevel doneTreeBuild()\n", thisIndex);
    }
#endif
    CkCallback cb(CkIndex_ParticleChunk::doneTreeBuild(), CkArrayIndex1D(0), chunks);
    contribute(0,0,CkReduction::concat,cb);
  }
#ifdef MEMCHECK
  CkPrintf("piece %d after doBuildTree()\n", thisIndex);
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
  done = 0;
  for(int i = 0; i < myParticles.length(); i++){
    //CkPrintf("[%d] inserting particle %d, current_root: %ld\n", thisIndex, myParticles[i]->num, NodeKey(Current_Root));
    (nodeptr) loadtree(myParticles[i], (cellptr) Current_Root, ProcessId);
    //Current_Root = (nodeptr) loadtree(myParticles[i], (cellptr) Current_Root, ProcessId);
    done++;
  }

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
  intcoord(xp, Pos(p),rmin,rsize);
  valid_root = TRUE;
  mynode = (nodeptr) root;
  kidIndex = subindex(xp, Level(mynode));
  qptr = &Subp(mynode)[kidIndex];

  l = Level(mynode) >> 1;

  flag = TRUE;
  while (flag) {                           /* loop descending tree     */
    if (l == 0) {
      //CkPrintf("[%d] not enough levels in tree, %d done\n", thisIndex, done);
      CkAbort("tree depth\n");
    }
    if (*qptr == NULL) { 
        le = InitLeaf((cellptr) mynode, ProcessId);
        ParentOf(p) = (nodeptr) le;
        Level(p) = l;
        ChildNum(p) = le->num_bodies;
        ChildNum(le) = kidIndex;
        NodeKey(le) = (NodeKey(mynode) << NDIM) + kidIndex;
        //CkPrintf("[%d] new leaf at kidindex: %d, key: %ld\n", thisIndex, kidIndex, NodeKey(le));
        Bodyp(le)[le->num_bodies++] = p;

        *qptr = (nodeptr) le;
        flag = FALSE;
    }
    if (flag && *qptr && (Type(*qptr) == LEAF)) {
      /*   reached a "leaf"?      */
        le = (leafptr) *qptr;
        //CkPrintf("[%d] reached existing leaf at kidindex %d key: %ld\n", thisIndex, kidIndex, NodeKey(le));
        if (le->num_bodies == MAX_BODIES_PER_LEAF) {
          //CkPrintf("[%d] too many particles. splitting\n", thisIndex);
          *qptr = (nodeptr) SubdivideLeaf(le, (cellptr) mynode, l, ProcessId);
        }
        else {
          //CkPrintf("[%d] enough space\n", thisIndex);
          ParentOf(p) = (nodeptr) le;
          Level(p) = l;
          ChildNum(p) = le->num_bodies;
          Bodyp(le)[le->num_bodies++] = p;
          flag = FALSE;
        }
    }
    if (flag) {
      mynode = *qptr;
      kidIndex = subindex(xp, l);
      qptr = &Subp(*qptr)[kidIndex];  /* move down one level  */
      l = l >> 1;                            /* and test next bit    */
    }
  }
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

cellptr TreePiece::makecell(unsigned ProcessId)
{
   cellptr c;
   int i, Mycell;
    
   if (myncell == maxmycell) {
      CkPrintf("makecell: Proc %d needs more than %d cells; increase fcells\n", 
	    ProcessId,maxmycell);
      CkAbort("makecell\n");
   }
   Mycell = myncell++;
   c = ctab + Mycell;
   Type(c) = CELL;
   Done(c) = FALSE;
   Mass(c) = 0.0;
   for (i = 0; i < NSUB; i++) {
      Subp(c)[i] = NULL;
   }
   mycelltab.push_back(c);
   return (c);
}

/*
 * MAKELEAF: allocation routine for leaves.
 */

leafptr TreePiece::makeleaf(unsigned ProcessId)
{
   leafptr le;
   int i, Myleaf;
    
   if (mynleaf == maxmyleaf) {
      CkPrintf("makeleaf: Proc %d needs more than %d leaves; increase fleaves\n",
	    ProcessId,maxmyleaf);
      CkAbort("makeleaf\n");
   }
   Myleaf = mynleaf++;
   le = ltab + Myleaf;
   le->seqnum = ProcessId * maxmyleaf + Myleaf;
   Type(le) = LEAF;
   Done(le) = FALSE;
   Mass(le) = 0.0;
   le->num_bodies = 0;
   for (i = 0; i < MAX_BODIES_PER_LEAF; i++) {
      Bodyp(le)[i] = NULL;
   }
   myleaftab.push_back(le);
   return (le);
}

cellptr
TreePiece::SubdivideLeaf (leafptr le, cellptr parent_, unsigned int l, unsigned int ProcessId)
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
   NodeKey(c) = (NodeKey(parent_) << NDIM)+ChildNum(c);
   //CkPrintf("new cell key: %ld\n", NodeKey(c));
   /* do first particle separately, so we can reuse le */
   p = bodies[0];
   intcoord(xp, Pos(p),rmin,rsize);
   index = subindex(xp, l);
   Subp(c)[index] = (nodeptr) le;
   ChildNum(le) = index;
   ParentOf(le) = (nodeptr) c;
   NodeKey(le) = (NodeKey(c) << NDIM)+index; 
   //CkPrintf("existing leaf (gets particle %d) key: %ld\n", p->num, NodeKey(le));

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
      intcoord(xp, Pos(p),rmin,rsize);
      index = subindex(xp, l);
      if (!Subp(c)[index]) {
	 le = InitLeaf(c, ProcessId);
	 ChildNum(le) = index;
	 Subp(c)[index] = (nodeptr) le;
         NodeKey(le) = (NodeKey(c) << NDIM) + index;
         //CkPrintf("i: %d, created leaf index %d (gets particle %d) key: %ld\n", i, index, p->num, NodeKey(le));
      }
      else {
	 le = (leafptr) Subp(c)[index];
         //CkPrintf("i: %d, existing leaf index %d (gets particle %d) key: %ld\n", i, index, p->num, NodeKey(le));
      }
      ParentOf(p) = (nodeptr) le;
      ChildNum(p) = le->num_bodies;
      Level(p) = l >> 1;
      Bodyp(le)[le->num_bodies++] = p;
   }
   return c;
}

cellptr TreePiece::InitCell(cellptr parent_, unsigned ProcessId)
{
  cellptr c;
  int i, Mycell;

  c = makecell(ProcessId);
  c->next = NULL;
  c->prev = NULL;
  if (parent_ == NULL)
    Level(c) = IMAX >> 1;
  else
    Level(c) = Level(parent_) >> 1;
  ParentOf(c) = (nodeptr) parent_;
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

bool intcoord(int *xp, vector rp, vector rmin, real rsize)
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
   }
}

// called only treepiece has children, so that myRoot will
// be valid.
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
  }
}

void TreePiece::cleanup(CkCallback &cb_){

  // reset your parent's pointer to your topmost 
  // cell/leaf. every treepiece has a parent node

  Subp(parent)[whichChildAmI] = NULL;

  numTotalMsgs = -1;
  numRecvdMsgs = 0;
  haveChildren = false;
  haveCounts = false;
  for(int i = 0; i < NSUB; i++){
    sentTo[i] = 0;
    partsToChild[i].length() = 0;
    isPending[i] = false;
  }
  // treepieces need not have myRoots
  // since we begin inserting particles as 
  // children of the parent node. therefore,
  // unless more than one bucket was created 
  // during the treebuild process, or the node
  // was split because it was too fat, myRoot
  // will be NULL
  if(myRoot != NULL){
    for(int i = 0; i < NSUB; i++){
      Subp(myRoot)[i] = NULL;
    }
  }
  myParticles.length() = 0;
  myNumParticles = 0;
  pendingChildren = 0;
  mycelltab.length() = 0;
  myleaftab.length() = 0;
  myncell = 0;
  mynleaf = 0;
  haveParent = false;
  bufferedMsgs.length() = 0;
  wantToSplit = false;

  contribute(0,0,CkReduction::concat,cb_);
#ifdef MEMCHECK
  CkPrintf("piece %d after cleanup\n", thisIndex);
  CmiMemoryCheck();
#endif
}

