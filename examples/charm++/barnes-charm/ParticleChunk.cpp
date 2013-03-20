#include "barnes.h"

#include "barnes.decl.h"
ParticleChunk::ParticleChunk(int mleaf, int mcell){

  usesAtSync = true;

  nstep = 0;
  numMsgsToEachTp = new int[numTreePieces];
  particlesToTps.resize(numTreePieces);
  for(int i = 0; i < numTreePieces; i++){
    numMsgsToEachTp[i] = 0;
    particlesToTps[i].reserve(MAX_PARTICLES_PER_MSG);
  }

};

ParticleChunk::~ParticleChunk(){
  for(int i = 0; i < numTreePieces; i++){
    particlesToTps[i].free();
  }
}

void ParticleChunk::doAtSync(CkCallback &cb_){
  mainCb = cb_;
  AtSync();
}

void ParticleChunk::ResumeFromSync(){
  contribute(0,0,CkReduction::concat,mainCb);
}

void ParticleChunk::pup(PUP::er &p){
  if(p.isUnpacking()){
    particlesToTps.resize(numTreePieces);
    for(int i = 0; i < numTreePieces; i++){
      particlesToTps[i].reserve(MAX_PARTICLES_PER_MSG);
    }
  }

  p | mybodytab;
  p | mynbody;
  p | bodytab;
  p | nstep;
  PUParray(p, rmin, NDIM);
  p | rsize;
  p | mainCb;
  p | numMsgsToEachTp;

}

void ParticleChunk::SlaveStart(CmiUInt8 bodyptrstart_, CmiUInt8 bodystart_, CkCallback &cb_){
  unsigned int ProcessId;

  /* Get unique ProcessId */
  ProcessId = thisIndex;

  bodyptr *bodyptrstart = (bodyptr *)bodyptrstart_;
  bodyptr bodystart = (bodyptr)bodystart_;

  /* POSSIBLE ENHANCEMENT:  Here is where one might pin processes to
     processors to avoid migration */

  /* initialize mybodytabs */
  mybodytab = bodyptrstart + (maxmybody * ProcessId);
  bodytab = bodystart;

  /* note that every process has its own copy   */
  /* of mybodytab, which was initialized to the */
  /* beginning of the whole array by proc. 0    */
  /* before create                              */
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

/* 
 * FIND_MY_INITIAL_BODIES: puts into mybodytab the initial list of bodies 
 * assigned to the processor.  
 */

void ParticleChunk::find_my_initial_bodies(bodyptr btab, int nbody, unsigned int ProcessId)
{
  int Myindex;
  int equalbodies;
  int extra,offset,i;

  int NPROC = numParticleChunks;

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
}

void ParticleChunk::acceptRoot(CmiUInt8 root_, real rminx, real rminy, real rminz, real rs, CkCallback &mainCb_){
  mainCb = mainCb_;
  G_root = (cellptr) root_;
  rsize = rs;
  rmin[0] = rminx;
  rmin[1] = rminy;
  rmin[2] = rminz;

#ifdef VERBOSE_CHUNKS
  CkPrintf("[%d] acceptRoot 0x%x\n", thisIndex, G_root);
#endif
  CkCallback cb(CkIndex_ParticleChunk::stepsystemPartII(0), thisProxy);
  contribute(0,0,CkReduction::concat,cb);
}

void ParticleChunk::partition(CkCallback &cb){
  int ProcessId = thisIndex;
  HouseKeep(); 
  real Cavg = (real) Cost(G_root) / (real)NPROC ;
  workMin = (int) (Cavg * ProcessId);
  workMax = (int) (Cavg * (ProcessId + 1) + (ProcessId == (NPROC - 1)));
#ifdef VERBOSE_CHUNKS
  CkPrintf("[%d] cost(root) = %f, workMin: %f, workMax: %f\n", ProcessId, Cavg, workMin, workMax);
#endif

  mynbody = 0;
  find_my_bodies((nodeptr)G_root, 0, BRC_FUC, ProcessId);
#ifdef VERBOSE_CHUNKS
  CkPrintf("[%d] mynbody: %d\n", thisIndex, mynbody);
#endif
  contribute(0,0,CkReduction::concat,cb);
#ifdef MEMCHECK
  CkPrintf("[%d] memcheck after partition\n", thisIndex);
  CmiMemoryCheck();
#endif
}

void ParticleChunk::HouseKeep(){
  myn2bcalc = mynbccalc = myselfint = 0;
}

void ParticleChunk::find_my_bodies(nodeptr mycell, int work, int direction, unsigned ProcessId){
  int i;
  leafptr l;
  nodeptr qptr;

  if (Type(mycell) == LEAF) {
    l = (leafptr) mycell;      
    for (i = 0; i < l->num_bodies; i++) {                                                            
      if (work >= workMin-.1) {                                                  
        if((mynbody) > maxmybody) {                                             
          CkPrintf("[%d] find_my_bodies: needs more than %d bodies; increase fleaves (%f) . mynbody: %d\n",ProcessId, maxmybody, fleaves, mynbody); 
          CkAbort("fleaves\n");
        }    
        mybodytab[mynbody++] = Bodyp(l)[i];
      }                                                                                             
      work += Cost(Bodyp(l)[i]);                                                                    
      if (work >= workMax-.1) {                                                    
        break;
      }                                                                                             
    }                                                                                                
  }
  else {
    for(i = 0; (i < NSUB) && (work < (workMax - .1)); i++){                         
      qptr = Subp(mycell)[Child_Sequence[direction][i]];                                            
      if (qptr!=NULL) {                                                                             
        if ((work+Cost(qptr)) >= (workMin -.1)) {                                 
          find_my_bodies(qptr,work, Direction_Sequence[direction][i],                             
              ProcessId);                                                              
        }
        work += Cost(qptr);                                                                        
      } 
    }
  }  
}

void ParticleChunk::ComputeForces (CkCallback &cb)
{
   bodyptr p,*pp;
   vector acc1, dacc, dvel, vel1, dpos;
   unsigned ProcessId = thisIndex;

   for (pp = mybodytab; pp < mybodytab+mynbody; pp++) {  
      p = *pp;
      SETV(acc1, Acc(p));
      Cost(p)=0;
      //CkPrintf("forces for particle %d\n", p->num);
      hackgrav(p,ProcessId);
      myn2bcalc += myn2bterm; 
      mynbccalc += mynbcterm;
      if (skipself) {       /*   did we miss self-int?  */
	 myselfint++;        /*   count another goofup   */
      }
      if (nstep > 0) {
	 /*   use change in accel to make 2nd order correction to vel      */
	 SUBV(dacc, Acc(p), acc1);
	 MULVS(dvel, dacc, dthf);
	 ADDV(Vel(p), Vel(p), dvel);
      }
#ifdef OUTPUT_ACC
      p->n2b = myn2bterm;
      p->nbc = mynbcterm;
#endif
   }

   contribute(0,0,CkReduction::concat,cb);
#ifdef MEMCHECK
  CkPrintf("[%d] memcheck after calcforces\n", thisIndex);
  CmiMemoryCheck();
#endif
}

void ParticleChunk::stepsystemPartII(CkReductionMsg *msg){

    delete msg;

    unsigned int ProcessId = thisIndex;

    /* load bodies into tree   */
    maketree(ProcessId);
    flushParticles();
    doneSendingParticles();
}

/*
 * MAKETREE: initialize tree structure for hack force calculation.
 */

void ParticleChunk::maketree(unsigned int ProcessId)
{
   bodyptr p, *pp;

   Current_Root = (nodeptr) G_root;
   for (pp = mybodytab; pp < mybodytab+mynbody; pp++) {
      p = *pp;
      if (Mass(p) != 0.0) {
	 Current_Root = (nodeptr) loadtree(p, (cellptr) Current_Root, ProcessId);
      }
      else {
	 fprintf(stderr, "Process %d found body 0x%x to have zero mass\n",
		 ProcessId, p);	
      }
   }
}

/*
 * LOADTREE: descend tree and insert particle.
 */

nodeptr
ParticleChunk::loadtree(bodyptr p, cellptr root, unsigned ProcessId)
{
   int l, xq[NDIM], xp[NDIM], flag;
   int i, j, root_level;
   bool valid_root;
   int kidIndex;
   nodeptr *qptr, mynode;
   cellptr c;
   leafptr le;

   intcoord(xp, Pos(p), rmin, rsize);
   root = G_root;
   mynode = (nodeptr) root;
   kidIndex = subindex(xp, Level(mynode));

   l = Level(mynode) >> 1;

   int depth = log8floor(numTreePieces);
   int lowestLevel = Level(mynode) >> depth;
   int fact = NDIM;
   int whichTp = 0;
   int d = depth-1;

   for(int level = Level(mynode); level > lowestLevel; level >>= 1){
     kidIndex = subindex(xp, Level(mynode));
     mynode = Subp(mynode)[kidIndex];
     whichTp += kidIndex*(1<<(fact*(d)));
     d--;     
   }

   int howMany = particlesToTps[whichTp].push_back_v(p); 
   if(howMany == MAX_PARTICLES_PER_MSG-1){ // enough particles to send 
     sendParticlesToTp(whichTp);
   }
}

void ParticleChunk::flushParticles(){
  for(int tp = 0; tp < numTreePieces; tp++){
    sendParticlesToTp(tp); 
  }
}

void ParticleChunk::sendParticlesToTp(int tp){
  int len = particlesToTps[tp].length();
  if(len > 0){
#ifdef VERBOSE_CHUNKS
    CkPrintf("[%d] sending %d particles to piece %d\n", thisIndex, len, tp);
#endif
    ParticleMsg *msg = new (len) ParticleMsg(); 
    memcpy(msg->particles, particlesToTps[tp].getVec(), len*sizeof(bodyptr));
    msg->num = len; 
    particlesToTps[tp].length() = 0;
    pieces[tp].recvParticles(msg);
  }
}

void ParticleChunk::doneSendingParticles(){
}

void ParticleChunk::doneTreeBuild(){
#ifdef VERBOSE_CHUNKS
  CkPrintf("[%d] all pieces have completed buildTree()\n", thisIndex);
#endif
  mainCb.send();
}

void ParticleChunk::advance(CkCallback &cb_){
  /* advance my bodies */

  real minmax[NDIM*2];
  int i;                                      
  bodyptr p,*pp;                                                          
  vector acc1, dacc, dvel, vel1, dpos;

  mainCb = cb_;
  SETVS(minmax,1E99);
  SETVS((minmax+NDIM),-1E99);

  for (pp = mybodytab; pp < mybodytab+mynbody; pp++) {
    p = *pp;
    MULVS(dvel, Acc(p), dthf);
    ADDV(vel1, Vel(p), dvel);
    MULVS(dpos, vel1, dtime);
    ADDV(Pos(p), Pos(p), dpos);
    ADDV(Vel(p), vel1, dvel);

    for (i = 0; i < NDIM; i++) {
      if (Pos(p)[i]<minmax[i]) {
        minmax[i]=Pos(p)[i];
      }
      if (Pos(p)[i]>minmax[NDIM+i]) {
        minmax[NDIM+i]=Pos(p)[i] ;
      }
    }
  }

  //CkPrintf("chunk %d minmax: (%f,%f,%f) (%f,%f,%f)\n", thisIndex, minmax[0], minmax[1], minmax[2], minmax[3], minmax[4], minmax[5]);
  CkCallback cb(CkIndex_Main::recvGlobalSizes(NULL), mainChare);
  contribute(NDIM*2*sizeof(real), minmax, minmax_RealVectorType, cb);
#ifdef MEMCHECK
  CkPrintf("[%d] memcheck after advance\n", thisIndex);
  CmiMemoryCheck();
#endif
}

void ParticleChunk::cleanup(){
  nstep++;
  contribute(0,0,CkReduction::concat,mainCb);
}

void ParticleChunk::outputAccelerations(CkCallback &cb_){
  bodyptr *pp;
  bodyptr p; 
  int i;
#ifdef OUTPUT_ACC
  for (i = 0, pp = mybodytab; pp < mybodytab+mynbody; pp++, i++) {  
    p = *pp;
    real *xp = Pos(p);
    real *ap = Acc(p);
    //CkPrintf("[%d] %d: pos: (%f,%f,%f), acc: (%f,%f,%f), nbc: %d, n2b: %d\n", thisIndex, p->num, xp[0], xp[1], xp[2], ap[0], ap[1], ap[2], p->nbc, p->n2b);
    ckerr << p->num << ": pos: (" << xp[0] << "," << xp[1] << "," << xp[2] << "), acc: (" << ap[0] << "," << ap[1] << "," << ap[2] << "), nbc: " << p->nbc << ", n2b: " << p->n2b << endl;
    //CkPrintf("%d: pos: (%f,%f,%f), acc: (%f,%f,%f), nbc: %d, n2b: %d\n", p->num, xp[0], xp[1], xp[2], ap[0], ap[1], ap[2], p->nbc, p->n2b);

  }
#endif
  //CkPrintf("[%d] bc: %d bb: %d\n", thisIndex, mynbccalc, myn2bcalc);
  contribute(0,0,CkReduction::concat,cb_);
}

