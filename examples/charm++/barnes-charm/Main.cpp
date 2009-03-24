#include "barnes.h"

CProxy_Main mainChare;
CProxy_TreePiece pieces;
CProxy_ParticleChunk chunks;

int numTreePieces;
int numParticleChunks;

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

  // create top portion of global tree
  int depth = log8floor(numTreePieces);
  nodeptr = createTopLevelTree(depth);

  // create holders, treepieces
  CProxy_BlockMap myMap=CProxy_BlockMap::ckNew(); 
  CkArrayOptions opts(numTreePieces); 

  opts.setMap(myMap);
  CProxy_TreePiece treeProxy = CProxy_TreePiece::ckNew(numTreePieces,opts);
  pieces = treeProxy;

  myMap=CProxy_BlockMap::ckNew(); 
  CkArrayOptions optss(numParticleChunks); 

  optss.setMap(myMap);
  CProxy_TreePiece chunkProxy = CProxy_TreePiece::ckNew(numParticleChunks,optss);
  chunks = chunkProxy;

  // slavestart for chunks
  chunks.SlaveStart(bodystart, cellstart, leafstart);

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
#include "barnes.def.h"
