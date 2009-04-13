#include "barnes.h"

CkReduction::reducerType chunksToPiecesReducerType;

#include "barnes.decl.h"
ParticleChunk::ParticleChunk(int maxleaf, int maxcell, int numchunks){

  /*allocate leaf/cell space */
  int NPROC = numchunks;
  ctab = (cellptr) G_MALLOC((maxcell / NPROC) * sizeof(cell));
  ltab = (leafptr) G_MALLOC((maxleaf / NPROC) * sizeof(leaf));

  numMsgsToEachTp = new int[numTreePieces];
  particlesToTps.resize(numTreePieces);
  for(int i = 0; i < numTreePieces; i++){
    particlesToTps[i].reserve(PARTICLES_PER_MSG);
  }

};

void ParticleChunk::SlaveStart(bodyptr *bodystart, cellptr *cellstart, leafptr *leafstart, CkCallback &cb_){
  unsigned int ProcessId;

  /* Get unique ProcessId */
  ProcessId = thisIndex;

  /* POSSIBLE ENHANCEMENT:  Here is where one might pin processes to
     processors to avoid migration */

  /* initialize mybodytabs */
  mybodytab = bodystart + (maxmybody * ProcessId);
  /* note that every process has its own copy   */
  /* of mybodytab, which was initialized to the */
  /* beginning of the whole array by proc. 0    */
  /* before create                              */
  mycelltab = cellstart + (maxmycell * ProcessId);
  myleaftab = leafstart + (maxmyleaf * ProcessId);
  /* POSSIBLE ENHANCEMENT:  Here is where one might distribute the
     data across physically distributed memories as desired. 

     One way to do this is as follows:

     int i;

     if (ProcessId == 0) {
     for (i=0;i<NPROC;i++) {
     Place all addresses x such that 
     &(Local[i]) <= x < &(Local[i])+
     sizeof(struct local_memory) on node i
     Place all addresses x such that 
     &(Local[i].mybodytab) <= x < &(Local[i].mybodytab)+
     maxmybody * sizeof(bodyptr) - 1 on node i
     Place all addresses x such that 
     &(Local[i].mycelltab) <= x < &(Local[i].mycelltab)+
     maxmycell * sizeof(cellptr) - 1 on node i
     Place all addresses x such that 
     &(Local[i].myleaftab) <= x < &(Local[i].myleaftab)+
     maxmyleaf * sizeof(leafptr) - 1 on node i
     }
     }

     barrier(Global->Barstart,NPROC);

*/

  find_my_initial_bodies(bodytab, nbody, ProcessId);

  contribute(0,0,CkReduction::concat,cb_);
}

void ParticleChunk::startIteration(CkCallback &cb_){

  mainCb = cb_; 
  int ProcessId = thisIndex;
  stepsystem(ProcessId);
}

/* 
 * FIND_MY_INITIAL_BODIES: puts into mybodytab the initial list of bodies 
 * assigned to the processor.  
 */

void ParticleChunk::find_my_initial_bodies(btab, nbody, ProcessId)
bodyptr btab;
int nbody;
unsigned int ProcessId;
{
  int Myindex;
  int equalbodies;
  int extra,offset,i;

  int NPROC = numParticleChunks;
  int ProcessId = thisIndex;

  mynbody = nbody / NPROC;
  extra = nbody % NPROC;
  if (ProcessId < extra) {
    mynbody++;    
    offset = mynbody * ProcessId;
  }
  if (ProcessId >= extra) {
    offset = (mynbody+1)*extra + (ProcessId - extra)*mynbody; 
  }
  for (i=0; i < mynbody; i++) {
     mybodytab[i] = &(btab[offset+i]);
  }
  //BARRIER(Global->Barstart,NPROC);
}

/*
 * STEPSYSTEM: advance N-body system one time-step.
 */

void ParticleChunk::stepsystem (ProcessId)
   unsigned int ProcessId;
{
  /*
    int i;
    real Cavg;
    bodyptr p,*pp;
    vector acc1, dacc, dvel, vel1, dpos;
    unsigned int time;
    unsigned int trackstart, trackend;
    unsigned int partitionstart, partitionend;
    unsigned int treebuildstart, treebuildend;
    unsigned int forcecalcstart, forcecalcend;
    */

    if (nstep == 2) {
/* POSSIBLE ENHANCEMENT:  Here is where one might reset the
   statistics that one is measuring about the parallel execution */
    }

    if ((ProcessId == 0) && (nstep >= 2)) {
        //CLOCK(trackstart);
    }

    if (ProcessId == 0) {
      // init_root bcasts root to all chunks
      // in the associated entry method, 
      // chunks contribute to continue with stepsystemPartII
       init_root(ProcessId);
    }
    else {
       mynumcell = 0;
       mynumleaf = 0;
    }
}

void ParticleChunk::stepsystemPartII(CkReductionMsg *msg){

    delete msg;

    unsigned int ProcessId = thisIndex;

    if ((ProcessId == 0) && (nstep >= 2)) {
        //CLOCK(treebuildstart);
    }

    /* load bodies into tree   */
    maketree(ProcessId);
    flushParticles();
    doneSendingParticles();

    if ((ProcessId == 0) && (nstep >= 2)) {
        //CLOCK(treebuildend);
        //Global->treebuildtime += treebuildend - treebuildstart;
    }
    
    // instead of barrier inside maketree:
    CkCallback cb(CkIndex_TreePiece::stepsystemPartIIb(), thisProxy);
    // stepsystemPartIIb
    contribute(0,0,CkReduction::concat,cb);

}

/*
 * MAKETREE: initialize tree structure for hack force calculation.
 */

void ParticleChunk::maketree(ProcessId)
   unsigned ProcessId;
{
  // FIXME - keep track of numMsgsToEachTp[] here 
   bodyptr p, *pp;

   myncell = 0;
   mynleaf = 0;
   if (ProcessId == 0) {
      mycelltab[myncell++] = G_root; 
   }
   Current_Root = (nodeptr) G_root;
   for (pp = mybodytab; 
	pp < mybodytab+mynbody; pp++) {
      p = *pp;
      if (Mass(p) != 0.0) {
	 Current_Root = (nodeptr) loadtree(p, 
                                           (cellptr) Current_Root, 
				           ProcessId);
      }
      else {
	 fprintf(stderr, "Process %d found body %d to have zero mass\n",
		 ProcessId, (int) p);	
      }
   }

   //BARRIER(Global->Bartree,NPROC);
   hackcofm( 0, ProcessId );
   //BARRIER(Global->Barcom,NPROC);
}

/*
 * LOADTREE: descend tree and insert particle.
 */

nodeptr
ParticleChunk::loadtree(p, root, ProcessId)
   bodyptr p;                        /* body to load into tree */
   cellptr root;
   unsigned ProcessId;
{
   int l, xq[NDIM], xp[NDIM], xor[NDIM], flag;
   int i, j, root_level;
   bool valid_root;
   int kidIndex;
   volatile nodeptr *volatile qptr, mynode;
   cellptr c;
   leafptr le;

   intcoord(xp, Pos(p));
   /*
   valid_root = TRUE;
   for (i = 0; i < NDIM; i++) {
      xor[i] = xp[i] ^ Root_Coords[i];
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
   root = G_root;
   mynode = (nodeptr) root;
   kidIndex = subindex(xp, Level(mynode));
   qptr = &Subp(mynode)[kidIndex];
   int depth = log8floor(numTreePieces);

   l = Level(mynode) >> 1;

   int depth = log8floor(numTreePieces);
   int lowestLevel = Level(mynode) >> depth;
   int fact = NSUB/2;
   int whichTp = 0;
   int d = depth;

   for(int level = Level(mynode); level >= lowestLevel; level >>= 1){
     kidIndex = subindex(xp, Level(mynode));
     mynode = &Subp(mynode)[kidIndex];
     whichTp += kidIndex*(1<<(fact*(d-1)));
     d--;     
   }

   int howMany = particlesToTps[whichTp].push_back_v(p); 
   if(howMany == MAX_PARTICLES_PER_MSG-1){ // enough particles to send 
     sendParticlesToTp(whichTp);
   }

   // this part should be done by the treepieces themselves
   /*
   flag = TRUE;
   while (flag) {                          
      if (l == 0) {
	 error("not enough levels in tree\n");
      }
      if (*qptr == NULL) { 
	 ALOCK(CellLock->CL, ((cellptr) mynode)->seqnum % MAXLOCK);
	 if (*qptr == NULL) {
	    le = InitLeaf((cellptr) mynode, ProcessId);
	    Parent(p) = (nodeptr) le;
	    Level(p) = l;
	    ChildNum(p) = le->num_bodies;
	    ChildNum(le) = kidIndex;
	    Bodyp(le)[le->num_bodies++] = p;
	    *qptr = (nodeptr) le;
	    flag = FALSE;
	 }
	 AULOCK(CellLock->CL, ((cellptr) mynode)->seqnum % MAXLOCK);
      }
      if (flag && *qptr && (Type(*qptr) == LEAF)) {
	 ALOCK(CellLock->CL, ((cellptr) mynode)->seqnum % MAXLOCK);
	 if (Type(*qptr) == LEAF) {             
	    le = (leafptr) *qptr;
	    if (le->num_bodies == MAX_BODIES_PER_LEAF) {
	       *qptr = (nodeptr) SubdivideLeaf(le, (cellptr) mynode, l,
						  ProcessId);
	    }
	    else {
	       Parent(p) = (nodeptr) le;
	       Level(p) = l;
	       ChildNum(p) = le->num_bodies;
	       Bodyp(le)[le->num_bodies++] = p;
	       flag = FALSE;
	    }
	 }
	 AULOCK(CellLock->CL, ((cellptr) mynode)->seqnum % MAXLOCK);
      }
      if (flag) {
	 mynode = *qptr;
         kidIndex = subindex(xp, l);
	 qptr = &Subp(*qptr)[kidIndex];  
	 l = l >> 1;                            
      }
   }
   SETV(Local[ProcessId].Root_Coords, xp);
   return Parent((leafptr) *qptr);
   */
}

void ParticleChunk::flushParticles(){
  // send any remaining particles to their 
  // intended host treepieces
  for(int tp = 0; tp < numTreePieces; tp++){
    sendParticlesToTp(tp); 
  }
}

void ParticleChunk::sendParticlesToTp(int tp){
  int len = particlesToTps[tp].length();
  if(len > 0){
    ParticleMsg *msg = new (len) ParticleMsg(); 
    for(int i = 0; i < len; i++){
      bodyptr tmp =  particlesToTps[tp][i];
      msg->particles[i] = *tmp;
    }
    msg->num = len; 
    numMsgsToEachTp[tp]++;
    particlesToTps[tp].length() = 0;
    pieces[tp].recvParticles(msg);
  }
}

void ParticleChunk::doneSendingParticles(){
  // send counts to chunk 0
  CkCallback cb(CkIndex_ParticleChunk::recvTotalMsgCountsFromChunks(0), pieces);
  contribute(numTreePieces*sizeof(int), numMsgsToEachTp, CkReduction::sum_int, cb);   
}

/*
void ParticleChunk::recvTotalMsgCounts(CkReductionMsg *msg){
  int *counts = (int *)msg->getData();
  for(int i = 0; i < numTreePieces; i++){
    pieces[i].recvTotalMsgCountsFromChunks(counts[i]);
  }
  delete msg;
}
*/

void ParticleChunk::stepSystemPartIII(CkReductionMsg *msg){

    delete msg;
    unsigned int ProcessId = thisIndex;
    unsigned int NPROC = numParticleChunks;

    Housekeep(ProcessId);

    Cavg = (real) Cost(G_root) / (real)NPROC ;
    Local[ProcessId].workMin = (int) (Cavg * ProcessId);
    Local[ProcessId].workMax = (int) (Cavg * (ProcessId + 1)
				      + (ProcessId == (NPROC - 1)));

    if ((ProcessId == 0) && (Local[ProcessId].nstep >= 2)) {
        CLOCK(partitionstart);
    }

    Local[ProcessId].mynbody = 0;
    find_my_bodies(Global->G_root, 0, BRC_FUC, ProcessId );

/*     B*RRIER(Global->Barcom,NPROC); */
    if ((ProcessId == 0) && (Local[ProcessId].nstep >= 2)) {
        CLOCK(partitionend);
        Global->partitiontime += partitionend - partitionstart;
    }

    if ((ProcessId == 0) && (Local[ProcessId].nstep >= 2)) {
        CLOCK(forcecalcstart);
    }

    ComputeForces(ProcessId);

    if ((ProcessId == 0) && (Local[ProcessId].nstep >= 2)) {
        CLOCK(forcecalcend);
        Global->forcecalctime += forcecalcend - forcecalcstart;
    }

    /* advance my bodies */
    for (pp = Local[ProcessId].mybodytab;
	 pp < Local[ProcessId].mybodytab+Local[ProcessId].mynbody; pp++) {
       p = *pp;
       MULVS(dvel, Acc(p), dthf);              
       ADDV(vel1, Vel(p), dvel);               
       MULVS(dpos, vel1, dtime);               
       ADDV(Pos(p), Pos(p), dpos);             
       ADDV(Vel(p), vel1, dvel);               
        
       for (i = 0; i < NDIM; i++) {
          if (Pos(p)[i]<Local[ProcessId].min[i]) {
	     Local[ProcessId].min[i]=Pos(p)[i];
	  }
          if (Pos(p)[i]>Local[ProcessId].max[i]) {
	     Local[ProcessId].max[i]=Pos(p)[i] ;
	  }
       }
    }
    LOCK(Global->CountLock);
    for (i = 0; i < NDIM; i++) {
       if (Global->min[i] > Local[ProcessId].min[i]) {
	  Global->min[i] = Local[ProcessId].min[i];
       }
       if (Global->max[i] < Local[ProcessId].max[i]) {
	  Global->max[i] = Local[ProcessId].max[i];
       }
    }
    UNLOCK(Global->CountLock);

    /* bar needed to make sure that every process has computed its min */
    /* and max coordinates, and has accumulated them into the global   */
    /* min and max, before the new dimensions are computed	       */
    BARRIER(Global->Barpos,NPROC);

    if ((ProcessId == 0) && (Local[ProcessId].nstep >= 2)) {
        CLOCK(trackend);
        Global->tracktime += trackend - trackstart;
    }
    if (ProcessId==0) {
      Global->rsize=0;
      SUBV(Global->max,Global->max,Global->min);
      for (i = 0; i < NDIM; i++) {
	if (Global->rsize < Global->max[i]) {
	   Global->rsize = Global->max[i];
	}
      }
      ADDVS(Global->rmin,Global->min,-Global->rsize/100000.0);
      Global->rsize = 1.00002*Global->rsize;
      SETVS(Global->min,1E99);
      SETVS(Global->max,-1E99);
    }
    Local[ProcessId].nstep++;
    Local[ProcessId].tnow = Local[ProcessId].tnow + dtime;
}

/*
 * INIT_ROOT: Processor 0 reinitialize the global root at each time step
 */
void ParticleChunk::init_root (ProcessId)
   unsigned int ProcessId;
{

  // create top portion of global tree
  int depth = log8floor(numTreePieces);
  //Global->G_root=Local[0].ctab;
  G_root = ctab[0];
  mynumcell=1;

  Type(G_root) = CELL;
  Done(G_root) = FALSE;
  Level(G_root) = IMAX >> 1;
  
  createTopLevelTree(G_root, depth);
  // send root to everyone
  chunks.acceptRoot(G_root);
  /*
  for (i = 0; i < NSUB; i++) {
    Subp(Global->G_root)[i] = NULL;
  }
  */
}

void ParticleChunk::createTopLevelTree(cellptr node, int depth){
  if(depth == 1){
    return;
  }
  
  for (int i = 0; i < NSUB; i++){
    cellptr child = makecell(thisIndex);
    // all code to initialize nodes goes here
    Subp(node)[i] = (nodeptr) child;
    Parent(child) = node;
    ChildNum(child) = i;
    createTopLevelTree(&children[i], depth-1);
  }
}

/*
 *  * MAKECELL: allocation routine for cells.
 */

cellptr ParticleChunk::makecell(ProcessId)
   unsigned ProcessId;
{
  cellptr c;
  int i, Mycell;

  if (mynumcell == maxmycell) {
    error("makecell: Proc %d needs more than %d cells; increase fcells\n", 
        ProcessId,maxmycell);
  }
  Mycell = mynumcell++;
  c = ctab + Mycell;
  c->seqnum = ProcessId*maxmycell+Mycell;
  Type(c) = CELL;
  Done(c) = FALSE;
  Mass(c) = 0.0;
  for (i = 0; i < NSUB; i++) {
    Subp(c)[i] = NULL;
  }
  mycelltab[myncell++] = c;
  return (c);
}

void ParticleChunk::acceptRoot(cellptr root_){
  G_root = root_;
  CkCallback cb(CkIndex_ParticleChunk::stepsystemPartII(0), thisProxy);
  contribute(0,0,CkReduction::concat,cb);
}

void registerChunksToPiecesReducer(){
  chunksToPiecesReducerType = CkReduction::addReducer(chunksToPiecesReducer);
}

CkReductionMsg *chunksToPiecesReducer (int nmsg, CkReductionMsg **msgs){
}
