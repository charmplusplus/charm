#include "barnes.h"

CProxy_Main mainChare;
CProxy_TreePiece pieces;
CProxy_ParticleChunk chunks;

int numTreePieces;
int numParticleChunks;

// number of particle chunks 
int NPROC;


#include "barnes.decl.h"

int log8floor(int arg){
  int ret = -1;
  for(; arg > 0; arg >>= 3){
    ret++;
  }
  return (ret);
}

Main::Main(CkArgMsg *m){
  int c;

  while ((c = getopt(m->argc, m->argv, "h")) != -1) {
    switch(c) {
      case 'h': 
        Help(); 
        exit(-1); 
        break;
      default:
        fprintf(stderr, "Only valid option is \"-h\".\n");
        exit(-1);
        break;
    }
  }

  // parameters
  initparam(m->argv, defv);
  startrun();
  initoutput();
  tab_init();

  numParticleChunks = NPROC;
  // various maximum count parameters are:
  // nbody, maxmybody, maxcell, maxmycell, maxleaf, maxmyleaf
  // nbody has already been set
  maxleaf = (int) ((double) fleaves * nbody);
  maxmyleaf = maxleaf/NPROC;
  maxcell = fcells * maxleaf;
  maxmycell = maxcell/NPROC;


  // FIXME - get numTreePieces from arguments
  // create chunks, treepieces
  CProxy_BlockMap myMap=CProxy_BlockMap::ckNew(); 
  CkArrayOptions opts(numTreePieces); 

  opts.setMap(myMap);
  CProxy_TreePiece treeProxy = CProxy_TreePiece::ckNew(numTreePieces,opts);
  pieces = treeProxy;

  myMap=CProxy_BlockMap::ckNew(); 
  CkArrayOptions optss(numParticleChunks); 

  optss.setMap(myMap);
  CProxy_TreePiece chunkProxy = CProxy_TreePiece::ckNew(maxleaf, maxcell, numParticleChunks, optss);
  chunks = chunkProxy;

  // startup split into two so that global readonly's
  // are initialized before we send start signal to 
  // particle chunks
  thisProxy.startSimulation();
}

void Main::startSimulation(){
  // slavestart for chunks

  chunks.SlaveStart(mybodytab, mycelltab, myleaftab, CkCallbackResumeThread());

  /* main loop */
  while (tnow < tstop + 0.1 * dtime) {
    chunks.startIteration(CkCallbackResumeThread());
  }
}

/*
 * TAB_INIT : allocate body and cell data space
 */
void 
Main::tab_init()
{

  /*allocate space for personal lists of body pointers */
  maxmybody = (nbody+maxleaf*MAX_BODIES_PER_LEAF)/NPROC; 
  mybodytab = (bodyptr*) G_MALLOC(NPROC*maxmybody*sizeof(bodyptr));
  /* space is allocated so that every */
  /* process can have a maximum of maxmybody pointers to bodies */ 
  /* then there is an array of bodies called bodytab which is  */
  /* allocated in the distribution generation or when the distr. */
  /* file is read */
  maxmycell = maxcell / NPROC;
  maxmyleaf = maxleaf / NPROC;
  mycelltab = (cellptr*) G_MALLOC(NPROC*maxmycell*sizeof(cellptr));
  myleaftab = (leafptr*) G_MALLOC(NPROC*maxmyleaf*sizeof(leafptr));

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
   string getparam();
   int getiparam();
   bool getbparam();
   double getdparam();
   int seed;

   infile = getparam("in");
   if (*infile != NULL) {
      inputdata();
   }
   else {
      nbody = getiparam("nbody");
      if (nbody < 1) {
	 error("startrun: absurd nbody\n");
      }
      seed = getiparam("seed");
   }

   outfile = getparam("out");
   dtime = getdparam("dtime");
   dthf = 0.5 * dtime;
   eps = getdparam("eps");
   epssq = eps*eps;
   tol = getdparam("tol");
   tolsq = tol*tol;
   fcells = getdparam("fcells");
   fleaves = getdparam("fleaves");
   tstop = getdparam("tstop");
   dtout = getdparam("dtout");
   NPROC = getiparam("NPROC");
   nstep = 0;
   pranset(seed);
   testdata();
   setbound();
   tout = tnow + dtout;
}

/*
 * TESTDATA: generate Plummer model initial conditions for test runs,
 * scaled to units such that M = -4E = G = 1 (Henon, Hegge, etc).
 * See Aarseth, SJ, Henon, M, & Wielen, R (1974) Astr & Ap, 37, 183.
 */

#define MFRAC  0.999                /* mass cut off at MFRAC of total */

void Main::testdata()
{
   real rsc, vsc, sqrt(), xrand(), pow(), rsq, r, v, x, y;
   vector cmr, cmv;
   register bodyptr p;
   int rejects = 0;
   int k;
   int halfnbody, i;
   float offset;
   register bodyptr cp;
   double tmp;

   headline = "Hack code: Plummer model";
   tnow = 0.0;
   bodytab = (bodyptr) G_MALLOC(nbody * sizeof(body));
   if (bodytab == NULL) {
      error("testdata: not enuf memory\n");
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

pickshell(vec, rad)
   real vec[];                     /* coordinate vector chosen */
   real rad;                       /* radius of chosen point */
{
   register int k;
   double rsq, xrand(), sqrt(), rsc;

   do {
      for (k = 0; k < NDIM; k++) {
	 vec[k] = xrand(-1.0, 1.0);
      }
      DOTVP(rsq, vec, vec);
   } while (rsq > 1.0);

   rsc = rad / sqrt(rsq);
   MULVS(vec, vec, rsc);
}


#include "barnes.def.h"
