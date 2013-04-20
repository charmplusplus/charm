#include "barnes.h"
#include "defs.h"

using std::string;

CProxy_Main mainChare;
CProxy_TreePiece pieces;
CProxy_ParticleChunk chunks;
int maxmybody;

int numTreePieces;
int maxPartsPerTp;
int numParticleChunks;

// number of particle chunks 
int NPROC;
real fcells;
real fleaves;
real tstop;
int nbody;
real dtime;
real eps;
real epssq;
real tol;
real tolsq;
real dtout;
real dthf;

int maxmycell;
int maxmyleaf;

CkReduction::reducerType minmax_RealVectorType;

int log8floor(int arg){
  int ret = -1;
  for(; arg > 0; arg >>= 3){
    ret++;
  }
  return (ret);
}

/*
 * INITOUTPUT: initialize output routines.
 */


void Main::initoutput()
{
   printf("\n\t\t%s\n\n", headline.c_str());
   printf("%10s%10s%10s%10s%10s%10s%10s%10s%10s\n",
	  "nbody", "dtime", "eps", "tol", "dtout", "tstop","fcells","fleaves","NPROC");
   printf("%10d%10.5f%10.4f%10.2f%10.3f%10.3f%10.2f%10.2f%10d\n\n",
	  nbody, dtime, eps, tol, dtout, tstop, fcells, fleaves, NPROC);
}


Main::Main(CkArgMsg *m){
  int c;

  mainChare = thisProxy;

  defaults.reserve(32);
  for(int i = 0; i < m->argc; i++){
    defaults.push_back(m->argv[i]);
  }
  startrun();
  initoutput();
  maxleaf = (int) ((double) fleaves * nbody);
  CkPrintf("[main] maxleaf: %d, fleaves: %f, nbody: %d\n", maxleaf, fleaves, nbody);
  tab_init();

  maxPartsPerTp = getiparam("fat"); 
  if(maxPartsPerTp < 0){
    maxPartsPerTp = MAX_PARTS_PER_TP;
  }
  else if(maxPartsPerTp < MAX_BODIES_PER_LEAF){
    maxPartsPerTp = MAX_BODIES_PER_LEAF;
  }

  
  numParticleChunks = NPROC;
  iterations = getiparam("it");
  if(iterations < 0){
    iterations = DEFAULT_NUM_ITERATIONS;
  }

  /*
  numTreePieces = getiparam("pieces");
  if(numTreePieces < 0){
    numTreePieces = 8*numParticleChunks; 
  }
  */
  int depth = getiparam("depth");
  if(depth < 1){
    depth = 1;
  }

  numTreePieces = 1 << (3*depth);
  ckout << "pieces: " << numTreePieces << endl;
  // various maximum count parameters are:
  // nbody, maxmybody, maxcell, maxmycell, maxleaf, maxmyleaf
  // nbody has already been set
  maxmyleaf = maxleaf/NPROC;
  maxcell = fcells * maxleaf;
  maxmycell = maxcell/NPROC;


  // create chunks, treepieces
  CProxy_BlockMap myMap=CProxy_BlockMap::ckNew(); 
  CkArrayOptions opts(numTreePieces); 
  opts.setMap(myMap);
  //int depth = log8floor(numTreePieces);
  ckout << "top-level pieces depth: " << (depth+1) << ", " << (IMAX >> (depth+1)) << endl;
  CProxy_TreePiece treeProxy = CProxy_TreePiece::ckNew((CmiUInt8)0,-1,(IMAX >> (depth+1)), CkArrayIndex1D(0), opts);
  pieces = treeProxy;

  myMap=CProxy_BlockMap::ckNew(); 
  CkArrayOptions optss(numParticleChunks); 
  optss.setMap(myMap);
  CProxy_ParticleChunk chunkProxy = CProxy_ParticleChunk::ckNew(maxleaf, maxcell, optss);
  chunks = chunkProxy;

  topLevelRoots.reserve(numTreePieces);
  // startup split into two so that global readonlys
  // are initialized before we send start signal to 
  // particle chunks
  ckout << "Starting simulation" << endl;
  thisProxy.startSimulation();
}

/*
 * MAKECELL: allocation routine for cells.
 */

cellptr Main::makecell(unsigned ProcessId)
{
   cellptr c;
   int i, Mycell;
    
   c = new cell;
   Type(c) = CELL;
   Done(c) = FALSE;
   Mass(c) = 0.0;
   for (i = 0; i < NSUB; i++) {
      Subp(c)[i] = NULL;
   }
   return (c);
}

/*
 * INIT_ROOT: Processor 0 reinitialize the global root at each time step
 */
void Main::init_root (unsigned int ProcessId)
{

  // create top portion of global tree
  int depth = log8floor(numTreePieces);
  G_root = makecell(ProcessId);
  ParentOf(G_root) = NULL;
  Mass(G_root) = 0.0;
  Cost(G_root) = 0;
  CLRV(Pos(G_root));
  NodeKey(G_root) = nodekey(1); 


  Level(G_root) = IMAX >> 1;
  
  CkPrintf("[main] Creating top-level tree, depth: %d, root: 0x%x\n", depth, G_root);
  int totalNumCellsMade = createTopLevelTree(G_root, depth);
  CkPrintf("totalNumCellsMade: %d\n", totalNumCellsMade+1);
}

int Main::createTopLevelTree(cellptr node, int depth){
  if(depth == 1){
    // lowest level, no more children will be created - save self
    int index = topLevelRoots.push_back_v((nodeptr) node);
#ifdef VERBOSE_MAIN
    CkPrintf("saving root %d 0x%x\n", index, node);
#endif
    return 0;
  }
  
  int numCellsMade = 0;
  // here, ancestors of treepieces' roots are created,
  // not the roots themselves.
  // these are all cells. the roots will start out as NULL pointers,
  // but if bodies are sent to them, they are modified to become
  // leaves.

  for (int i = 0; i < NSUB; i++){
    cellptr child = makecell(-1);
    Subp(node)[i] = (nodeptr) child;
    ParentOf(child) = (nodeptr) node;
    ChildNum(child) = i;
    Level(child) = Level(node) >> 1;
    nodekey k = NodeKey(node) << NDIM;
    k += i;
    NodeKey(child) = k;

    Mass(child) = 0.0;
    Cost(child) = 0;
    CLRV(Pos(child));

    numCellsMade++;
    numCellsMade += createTopLevelTree((cellptr) Subp(node)[i], depth-1);
  }

  return numCellsMade;
}

void Main::startSimulation(){
  // slavestart for chunks
  chunks.SlaveStart((CmiUInt8)mybodytab, (CmiUInt8)bodytab, CkCallbackResumeThread());

  double start;
  double end;
  double totalStart;
  double iterationStart;
  double totalEnd;

  /* main loop */
  int i = 0;
  while (tnow < tstop + 0.1 * dtime && i < iterations) {
    // create top-level tree
    CkPrintf("**********************************\n");
    CkPrintf("[main] iteration: %d, tnow: %f\n", i, tnow);
    CkPrintf("[main] rmin: (%f,%f,%f), rsize: %f\n", rmin[0], rmin[1], rmin[2], rsize);
    CkPrintf("**********************************\n");
#ifndef NO_TIME
    start = CkWallTimer();
    iterationStart = CkWallTimer();
#endif
    if(i == 2){
      totalStart = CkWallTimer();
    }
    CkCallback cb(CkIndex_TreePiece::doBuildTree(), pieces);
    CkStartQD(cb);
    init_root(-1);
    // send roots to pieces
    pieces.acceptRoots((CmiUInt8)topLevelRoots.getVec(), rsize, rmin[0], rmin[1], rmin[2], CkCallbackResumeThread());
    // send root to chunk
    chunks.acceptRoot((CmiUInt8)G_root, rmin[0], rmin[1], rmin[2], rsize, CkCallbackResumeThread());
    
    // update top-level nodes' moments here, since all treepieces have 
    // completed calculating theirs
    updateTopLevelMoments();
#ifndef NO_TIME
    end = CkWallTimer();
#endif
    CkPrintf("[main] Tree building ... %f s\n", (end-start));
#ifdef PRINT_TREE
    graph();
#endif
#ifndef NO_PARTITION
#ifndef NO_TIME
    start = CkWallTimer();
#endif
    chunks.partition(CkCallbackResumeThread());
#ifndef NO_TIME
    end = CkWallTimer();
#endif
    CkPrintf("[main] Partitioning ...  %f s\n", (end-start));
#endif
#ifndef NO_FORCES
#ifndef NO_TIME
    start = CkWallTimer();
#endif
    chunks.ComputeForces(CkCallbackResumeThread());
#ifndef NO_TIME
    end = CkWallTimer();
#endif
    CkPrintf("[main] Forces ...  %f s\n", (end-start));
#endif
#ifndef NO_ADVANCE
#ifndef NO_TIME
    start = CkWallTimer();
#endif
    chunks.advance(CkCallbackResumeThread());
#ifndef NO_TIME
    end = CkWallTimer();
#endif
    CkPrintf("[main] Advance ... %f s\n", (end-start));
#endif
#ifndef NO_CLEANUP
#ifndef NO_TIME
    start = CkWallTimer();
#endif
    pieces.cleanup(CkCallbackResumeThread());
#ifndef NO_TIME
    end = CkWallTimer();
#endif
    CkPrintf("[main] Clean up ... %f s\n", (end-start));
#endif
    i++;
#ifndef NO_TIME
    totalEnd = CkWallTimer();
#endif
    CkPrintf("[main] Total ... %f s\n", (totalEnd-iterationStart));
#ifndef NO_LB
    CkPrintf("[main] starting LB\n");
    chunks.doAtSync(CkCallbackResumeThread());
#endif

    // must reset the vector of top level roots so that the same
    // root isn't used over and over again by the treepieces:
    topLevelRoots.length() = 0;
    tnow = tnow + dtime;
  }
  totalEnd = CkWallTimer();

  CkPrintf("[main] Completed simulation: %f s\n", (totalEnd-totalStart));
#ifdef OUTPUT_ACC
  chunks.outputAccelerations(CkCallbackResumeThread());
#endif
  CkExit();
}



/*
 * TAB_INIT : allocate body and cell data space
 */
void 
Main::tab_init()
{

  /*allocate space for personal lists of body pointers */
  maxmybody = (real)(nbody+maxleaf*MAX_BODIES_PER_LEAF)/(real) NPROC; 
  CkPrintf("[main] maxmybody: %d, nbody: %d, maxleaf: %d, MBPL: %d, fleaves: %f\n", maxmybody, nbody, maxleaf, MAX_BODIES_PER_LEAF, fleaves);
  mybodytab = new bodyptr [NPROC*maxmybody]; 
  CkAssert(mybodytab != NULL);
  /* space is allocated so that every */
  /* process can have a maximum of maxmybody pointers to bodies */ 
  /* then there is an array of bodies called bodytab which is  */
  /* allocated in the distribution generation or when the distr. */
  /* file is read */
  maxmycell = maxcell / NPROC;
  maxmyleaf = maxleaf / NPROC;

}


void
Main::Help () 
{
   printf("There are a total of twelve parameters, and all of them have default values.\n");
   printf("\n");
   printf("1) infile (char*) : The name of an input file that contains particle data.  \n");
   printf("    The  of the file is:\n");
   printf("\ta) An int representing the number of particles in the distribution\n");
   printf("\tb) An int representing the dimensionality of the problem (3-D)\n");
   printf("\tc) A double representing the current time of the simulation\n");
   printf("\td) Doubles representing the masses of all the particles\n");
   printf("\te) A vector (length equal to the dimensionality) of doubles\n");
   printf("\t   representing the positions of all the particles\n");
   printf("\tf) A vector (length equal to the dimensionality) of doubles\n");
   printf("\t   representing the velocities of all the particles\n");
   printf("\n");
   printf("    Each of these numbers can be separated by any amount of whitespace.\n");
   printf("\n");
   printf("2) nbody (int) : If no input file is specified (the first line is blank), this\n");
   printf("    number specifies the number of particles to generate under a plummer model.\n");
   printf("    Default is 16384.\n");
   printf("\n");
   printf("3) seed (int) : The seed used by the random number generator.\n");
   printf("    Default is 123.\n");
   printf("\n");
   printf("4) outfile (char*) : The name of the file that snapshots will be printed to. \n");
   printf("    This feature has been disabled in the SPLASH release.\n");
   printf("    Default is NULL.\n");
   printf("\n");
   printf("5) dtime (double) : The integration time-step.\n");
   printf("    Default is 0.025.\n");
   printf("\n");
   printf("6) eps (double) : The usual potential softening\n");
   printf("    Default is 0.05.\n");
   printf("\n");
   printf("7) tol (double) : The cell subdivision tolerance.\n");
   printf("    Default is 1.0.\n");
   printf("\n");
   printf("8) fcells (double) : The total number of cells created is equal to \n");
   printf("    fcells * number of leaves.\n");
   printf("    Default is 2.0.\n");
   printf("\n");
   printf("9) fleaves (double) : The total number of leaves created is equal to  \n");
   printf("    fleaves * nbody.\n");
   printf("    Default is 0.5.\n");
   printf("\n");
   printf("10) tstop (double) : The time to stop integration.\n");
   printf("    Default is 0.075.\n");
   printf("\n");
   printf("11) dtout (double) : The data-output interval.\n");
   printf("    Default is 0.25.\n");
   printf("\n");
   printf("12) NPROC (int) : The number of processors.\n");
   printf("    Default is 1.\n");
}

/*
 * STARTRUN: startup hierarchical N-body code.
 */

void Main::startrun()
{
   int seed;

   infile = getparam("in");
   if (!infile.empty()) {
     CkPrintf("[main] input file: %s\n", infile.c_str());
     inputdata();
   }
   else {
      nbody = getiparam("nbody");
      if (nbody < 1) {
	 ckerr << "startrun: absurd nbody\n";
         nbody = 16384;
      }
      seed = getiparam("seed");
      if(seed < 0)
        seed = 123;
   }

   outfile = getparam("out");
   dtime = getdparam("dtime");
   if(isnan(dtime))
     dtime = 0.025;

   dthf = 0.5 * dtime;
   eps = getdparam("eps");
   if(isnan(eps))
     eps = 0.05;
   epssq = eps * eps;

   real epssq = eps*eps;
   tol = getdparam("tol");
   if(isnan(tol))
     tol = 1.0;
   tolsq = tol*tol;
   
   fcells = getdparam("fcells");
   if(isnan(fcells))
     fcells = 2.0;

   fleaves = getdparam("fleaves");
   if(isnan(fleaves))
     fleaves = 15;

   tstop = getdparam("tstop");
   if(isnan(tstop))
     tstop = 0.225;

   dtout = getdparam("dtout");
   if(isnan(dtout))
     dtout = 0.25;

   NPROC = getiparam("NPROC");
   if(NPROC < 0)
     NPROC = 1;

   pranset(seed);

   if(infile.empty()){
    testdata();
   }
   setbound();
   tout = tnow + dtout;
}

/*
 * SETBOUND: Compute the initial size of the root of the tree; only done
 * before first time step, and only processor 0 does it
 */
void Main::setbound()
{
   int i;
   real side;
   bodyptr p;
   vector min, max;

   SETVS(min,1E99);
   SETVS(max,-1E99);
   side=0;

   for (p = bodytab; p < bodytab+nbody; p++) {
      for (i=0; i<NDIM;i++) {
	 if (Pos(p)[i]<min[i]) min[i]=Pos(p)[i] ;
	 if (Pos(p)[i]>max[i])  max[i]=Pos(p)[i] ;
      }
   }
    
   SUBV(max,max,min);
   for (i=0; i<NDIM;i++) if (side<max[i]) side=max[i];
   ADDVS(rmin,min,-side/100000.0);
   rsize = 1.00002*side;
}

/*
 * TESTDATA: generate Plummer model initial conditions for test runs,
 * scaled to units such that M = -4E = G = 1 (Henon, Hegge, etc).
 * See Aarseth, SJ, Henon, M, & Wielen, R (1974) Astr & Ap, 37, 183.
 */

#define MFRAC  0.999                /* mass cut off at MFRAC of total */

void Main::testdata()
{
   real rsc, vsc, rsq, r, v, x, y;
   vector cmr, cmv;
   register bodyptr p;
   int rejects = 0;
   int k;
   int halfnbody, i;
   float offset;
   register bodyptr cp;
   double tmp;

   tnow = 0.0;
   bodytab = new body [nbody];
   CkAssert(bodytab);
   if (bodytab == NULL) {
      ckerr << "testdata: not enuf memory\n";
   }
   rsc = 9 * PI / 16;
   vsc = sqrt(1.0 / rsc);

   CLRV(cmr);
   CLRV(cmv);

   halfnbody = nbody / 2;
   if (nbody % 2 != 0) halfnbody++;
   for (p = bodytab; p < bodytab+halfnbody; p++) {
      Type(p) = BODY;
      Mass(p) = 1.0 / nbody;
      Cost(p) = 1;

      r = 1 / sqrt(pow(xrand(0.0, MFRAC), -2.0/3.0) - 1);
      /*   reject radii greater than 10 */
      while (r > 9.0) {
	 rejects++;
	 r = 1 / sqrt(pow(xrand(0.0, MFRAC), -2.0/3.0) - 1);
      }        
      pickshell(Pos(p), rsc * r);
      ADDV(cmr, cmr, Pos(p));
      do {
	 x = xrand(0.0, 1.0);
	 y = xrand(0.0, 0.1);

      } while (y > x*x * pow(1 - x*x, 3.5));

      v = sqrt(2.0) * x / pow(1 + r*r, 0.25);
      pickshell(Vel(p), vsc * v);
      ADDV(cmv, cmv, Vel(p));
   }

   offset = 4.0;

   for (p = bodytab + halfnbody; p < bodytab+nbody; p++) {
      Type(p) = BODY;
      Mass(p) = 1.0 / nbody;
      Cost(p) = 1;

      cp = p - halfnbody;
      for (i = 0; i < NDIM; i++){
	 Pos(p)[i] = Pos(cp)[i] + offset; 
	 ADDV(cmr, cmr, Pos(p));
	 Vel(p)[i] = Vel(cp)[i];
	 ADDV(cmv, cmv, Vel(p));
      }
   }

   DIVVS(cmr, cmr, (real) nbody);
   DIVVS(cmv, cmv, (real) nbody);

   for (p = bodytab; p < bodytab+nbody; p++) {
      SUBV(Pos(p), Pos(p), cmr);
      SUBV(Vel(p), Vel(p), cmv);
   }
}

/*
 * PICKSHELL: pick a random point on a sphere of specified radius.
 */

void Main::pickshell(real vec[], real rad)
{
   register int k;
   double rsq, rsc;

   do {
      for (k = 0; k < NDIM; k++) {
	 vec[k] = xrand(-1.0, 1.0);
      }
      DOTVP(rsq, vec, vec);
   } while (rsq > 1.0);

   rsc = rad / sqrt(rsq);
   MULVS(vec, vec, rsc);
}

/*
 * GETPARAM: export version prompts user for value.
 */

string Main::getparam(string name)
{
   int i, leng;
   string def;
   char* temp;

  for(int i = 0; i < defaults.length(); i++){
    if(defaults[i].find(name) != string::npos){
      int pos = defaults[i].find("=");
      string value = defaults[i].substr(pos+1,defaults[i].length()-pos-1);
      return value;
    }
  }
  ckout << "getparam: " << name.c_str() << " unknown\n";
  return string();
}

/*
 * GETIPARAM, ..., GETDPARAM: get int, long, bool, or double parameters.
 */

int Main::getiparam(string name)
{
  string val;

  val = getparam(name);
  if(val.empty())
    return -1;
  else
    return (atoi(val.c_str()));
}

long Main::getlparam(string name)
{
  string val;

  val = getparam(name);
  if(val.empty())
    return -1;
  else 
    return (atol(val.c_str()));
}

bool Main::getbparam(string name)
{
  string val;

  val = getparam(name);
  if (strchr("tTyY1", *(val.c_str())) != 0) {
    return (true);
  }
  if (strchr("fFnN0", *(val.c_str())) != 0) {
    return (false);
  }
  CkPrintf("getbparam: %s=%s not bool\n", name.c_str(), val.c_str());
}

double Main::getdparam(string name)
{
  string val;

  val = getparam(name);
  if(val.empty())
    return NAN;
  else 
    return (atof(val.c_str()));
}

/*
 * EXTRVALUE: extract value from name=value string.
 */

string Main::extrvalue(string &arg)
{
   char *ap;

   ap = (char *) arg.c_str();
   while (*ap != 0)
      if (*ap++ == '=')
	 return (string(ap));
   return (string());
}

void Main::updateTopLevelMoments(){
  int depth = log8floor(numTreePieces);
#ifdef VERBOSE_MAIN
  CkPrintf("[main]: updateTopLevelMoments(%d)\n", depth);
#endif
  moments((nodeptr)G_root, depth);
#ifdef MEMCHECK
  CkPrintf("[main] check after moments\n");
  CmiMemoryCheck();
#endif
}

nodeptr Main::moments(nodeptr node, int depth){
  vector tmpv;
  if(depth == 0){
    return node;
  }
  for(int i = 0; i < NSUB; i++){
    nodeptr child = Subp(node)[i];
    if(child != NULL){
      nodeptr mom = moments(child, depth-1);
#ifdef VERBOSE_MAIN
      CkPrintf("node 0x%x (%f,%f,%f) with node 0x%x (%d) (%f,%f,%f) mass: %f\n", node, Pos(node)[0], Pos(node)[1], Pos(node)[2], mom, i, Pos(mom)[0], Pos(mom)[1], Pos(mom)[2], Mass(mom));
#endif
      Mass(node) += Mass(mom);
      Cost(node) += Cost(mom);
      MULVS(tmpv, Pos(mom), Mass(mom));
      ADDV(Pos(node), Pos(node), tmpv);
    }
  }
  DIVVS(Pos(node), Pos(node), Mass(node));
#ifdef VERBOSE_MAIN
  CkPrintf("Pos(node): (%f,%f,%f)\n", Pos(node)[0], Pos(node)[1], Pos(node)[2]);
#endif
  return node;
}

#ifdef PRINT_TREE
void Main::graph(){
  ofstream myfile;
  ostringstream ostr;

  ostr << "tree." << nbody << "." << maxPartsPerTp << ".dot";
  CkPrintf("[main] output file name: %s\n", ostr.str().c_str());
  myfile.open(ostr.str().c_str());
  myfile << "digraph tree_" << nbody << "_" << maxPartsPerTp <<" {" << endl;
  CkQ<nodeptr> nodes(4096);
  nodes.enq((nodeptr)G_root);
  while(!nodes.isEmpty()){
    nodeptr curnode = nodes.deq();
    
    myfile << NodeKey(curnode) << " [label=\"" << "("<< NodeKey(curnode) << ", " << Mass(curnode) << ")" << "\\n (" << Pos(curnode)[0] << "," << Pos(curnode)[1] << "," << Pos(curnode)[2] << ") " << "\"];"<< endl;
    for(int i = 0; i < NSUB; i++){
      nodeptr childnode = Subp(curnode)[i];
      if(childnode != NULL){
        if(Type(childnode) == CELL){
          nodes.enq(childnode);
          myfile << NodeKey(curnode) << "->" << NodeKey(childnode) << endl;
        }
        else if(Type(childnode) == LEAF){
          myfile << NodeKey(curnode) << "->" << NodeKey(childnode) << endl;
          myfile << NodeKey(childnode) << " [label=\"" << "("<< NodeKey(childnode) << ", " << Mass(childnode) << ")" << "\\n (" << Pos(childnode)[0] << "," << Pos(childnode)[1] << "," << Pos(childnode)[2] << ") num " << (((leafptr)childnode)->bodyp[0])->num << " \"];"<< endl;
        }
      }
      else {
        myfile << NodeKey(curnode) << "-> NULL_" << NodeKey(curnode) << "_" << i << endl;
      }
    }
  }
  myfile << "}" << endl;
  myfile.close();
}
#endif

CkReductionMsg *minmax_RealVector(int nmsg, CkReductionMsg **msgs){
  real minmax[NDIM*2];

  SETVS(minmax,1E99);
  SETVS((minmax+NDIM),-1E99);

  for(int j = 0; j < nmsg; j++){
    CkAssert(msgs[j]->getSize() == NDIM*2*sizeof(real));
    real *tmp = (real *) msgs[j]->getData();
    real *min = tmp;
    real *max = tmp+NDIM;
    for (int i = 0; i < NDIM; i++) {
      if (minmax[i] > min[i]) {
        minmax[i] = min[i];
      }
      if (minmax[NDIM+i] < max[i]) {
        minmax[NDIM+i] = max[i];
      }
    }
  }

  return CkReductionMsg::buildNew(NDIM*2*sizeof(real), minmax);
}

void register_minmax_RealVector(){
  minmax_RealVectorType = CkReduction::addReducer(minmax_RealVector); 
}

void Main::recvGlobalSizes(CkReductionMsg *msg){
  real *tmp = (real *)msg->getData();
  real *min = tmp;
  real *max = tmp+NDIM;

  rsize=0;
  SUBV(max,max,min);
  for (int i = 0; i < NDIM; i++) {
    if (rsize < max[i]) {
      rsize = max[i];
    }
  }
  ADDVS(rmin,min,-rsize/100000.0);
  rsize = 1.00002*rsize;
  chunks.cleanup();

  delete msg;
}
#include "barnes.def.h"
