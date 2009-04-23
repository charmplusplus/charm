#include "barnes.h"
#include "defs.h"

CProxy_Main mainChare;
CProxy_TreePiece pieces;
CProxy_ParticleChunk chunks;
int maxmybody;

int numTreePieces;
int numParticleChunks;

// number of particle chunks 
int NPROC;
real fcells;
real fleaves;
real tstop;
int nbody;
real dtime;
real eps;
real tol;
real dtout;
real dthf;
vector rmin;
real rsize;



int log8floor(int arg){
  int ret = -1;
  for(; arg > 0; arg >>= 3){
    ret++;
  }
  return (ret);
}

/*
 * INITPARAM: ignore arg vector, remember defaults.
 */

/*
void Main::initparam(string *argv, const char **defvs)
{
   defaults = defvs;
}
*/

/*
 * INITOUTPUT: initialize output routines.
 */


void Main::initoutput()
{
   printf("\n\t\t%s\n\n", headline.c_str());
   printf("%10s%10s%10s%10s%10s%10s%10s%10s\n",
	  "nbody", "dtime", "eps", "tol", "dtout", "tstop","fcells","NPROC");
   printf("%10d%10.5f%10.4f%10.2f%10.3f%10.3f%10.2f%10d\n\n",
	  nbody, dtime, eps, tol, dtout, tstop, fcells, NPROC);
}



Main::Main(CkArgMsg *m){
  int c;

/*
  while ((c = getopt(m->argc, m->argv, "h")) != -1) {
    switch(c) {
      case 'h': 
        Help(); 
        exit(-1); 
        break;
      case 't':
        numTreePieces = c;
        break;
        
      default:
        fprintf(stderr, "Only valid options are \"-h\" and \"-t\"\n");
        exit(-1);
        break;
    }
  }
  */

  defaults.reserve(32);
  for(int i = 0; i < m->argc; i++){
    defaults.push_back(m->argv[i]);
  }
  /*
  defaults.push_back("in=");
  defaults.push_back("out=");
  defaults.push_back("nbody=16384");
  defaults.push_back("seed=123");
  defaults.push_back("dtime=0.025");
  defaults.push_back("eps=0.05");
  defaults.push_back("tol=1.0");
  defaults.push_back("fcells=2.0");
  defaults.push_back("fleaves=0.5");
  defaults.push_back("tstop=0.075");
  defaults.push_back("dtout=0.25");
  defaults.push_back("NPROC=1");
  */

  // parameters
  //initparam(m->argv, defv);
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
  // FIXME - level of top-level trees
  int depth = log8floor(numTreePieces);
  CProxy_TreePiece treeProxy = CProxy_TreePiece::ckNew((nodeptr)0,-1,true,(IMAX >> (depth+1)),opts);
  pieces = treeProxy;

  myMap=CProxy_BlockMap::ckNew(); 
  CkArrayOptions optss(numParticleChunks); 
  optss.setMap(myMap);
  CProxy_ParticleChunk chunkProxy = CProxy_ParticleChunk::ckNew(maxleaf, maxcell, optss);
  chunks = chunkProxy;

  // startup split into two so that global readonlys
  // are initialized before we send start signal to 
  // particle chunks
  thisProxy.startSimulation();
}

void Main::startSimulation(){
  // slavestart for chunks

  chunks.SlaveStart((CmiUInt8)mybodytab, (CmiUInt8)bodytab, CkCallbackResumeThread());
  //chunks.SlaveStart(mybodytab, mycelltab, myleaftab, CkCallbackResumeThread());

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
  // FIXME - this should be a member of ParticleChunk
  bodyptr *mybodytab = (bodyptr*) G_MALLOC(NPROC*maxmybody*sizeof(bodyptr));
  /* space is allocated so that every */
  /* process can have a maximum of maxmybody pointers to bodies */ 
  /* then there is an array of bodies called bodytab which is  */
  /* allocated in the distribution generation or when the distr. */
  /* file is read */
  maxmycell = maxcell / NPROC;
  maxmyleaf = maxleaf / NPROC;
  //mycelltab = (cellptr*) G_MALLOC(NPROC*maxmycell*sizeof(cellptr));
  //myleaftab = (leafptr*) G_MALLOC(NPROC*maxmyleaf*sizeof(leafptr));

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
   if (infile.empty()) {
     // FIXME - enable
      //inputdata();
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

   real epssq = eps*eps;
   tol = getdparam("tol");
   if(isnan(tol))
     tol = 1.0;

   real tolsq = tol*tol;
   fcells = getdparam("fcells");
   if(isnan(fcells))
     fcells = 2.0;

   fleaves = getdparam("fleaves");
   if(isnan(fleaves))
     fleaves = 0.5;

   tstop = getdparam("tstop");
   if(isnan(tstop))
     tstop = 0.075;

   dtout = getdparam("dtout");
   if(isnan(dtout))
     dtout = 0.25;

   NPROC = getiparam("NPROC");
   if(NPROC < 0)
     NPROC = 1;

   srand(seed);
   testdata();
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
   real side ;
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
   SETVS(max,-1E99);
   SETVS(min,1E99);
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

   //headline = "Hack code: Plummer model";
   tnow = 0.0;
   bodytab = (bodyptr) G_MALLOC(nbody * sizeof(body));
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
//   real vec[];                     /* coordinate vector chosen */
//   real rad;                       /* radius of chosen point */
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
   //char buf[128];
   char* temp;

  /*
   if (defaults == NULL)
      ckerr << "getparam: called before initparam\n";
    */
  for(int i = 0; i < defaults.length(); i++){
    if(defaults[i].find(name) != string::npos){
      int pos = defaults[i].find("=");
      string value = defaults[i].substr(pos+1,defaults[i].length()-pos-1);
      return value;
    }
  }
  ckerr << "getparam: " << name.c_str() << "unknown\n";
  return string();
  /*
   i = scanbind(defaults, name);
   if (i < 0)
      ckerr << "getparam: %s unknown\n", name;
   def = extrvalue(defaults[i]);
   gets(buf);
   leng = strlen(buf) + 1;
   if (leng > 1) {
      return (strcpy((char *)malloc(leng), buf));
   }
   else {
      return (def);
   }
   */
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
 * SCANBIND: scan binding vector for name, return index.
 */

/*
int Main::scanbind(CkVec<string> &bvec, string &name)
{
   int i;

   for(i = 0; i < bvec.length(); i++){
     if(matchname(bvec[i], name)){
       return i;
     }
   }
   return (-1);
}
*/

/*
 * MATCHNAME: determine if "name=value" matches "name".
 */

/*
bool Main::matchname(string &bind, string &name)
{
   char *bp, *np;

   bp = bind.c_str();
   np = name.c_str();
   while (*bp == *np) {
      bp++;
      np++;
   }
   return (*bp == '=' && *np == 0);
}
*/

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

real xrand(real lo, real hi){
  real ran = lo+hi*drand48();
  return ran;
}

#include "barnes.def.h"
