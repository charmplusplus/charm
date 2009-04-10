#ifndef _BARNES_H_
#define _BARNES_H_

#include "barnes.decl.h"
#define LOCK(x) /* empty */

#define MAX_PARTS_PER_TP 1000

typedef double real;

extern CProxy_Main mainChare;
extern CProxy_TreePiece pieces;
extern CProxy_ParticleChunk chunks;

extern int numTreePieces;
extern int numParticleChunks;
extern int nbody;
extern int NPROC;

extern real tstop;

class ParticleMsg : public CMessage_ParticleMsg {
  public:
  body *particles;
  int num;
};

class Main : public CBase_Main {

  char *defv[] = {                 /* DEFAULT PARAMETER VALUES              */
    /* file names for input/output                                         */
    "in=",                        /* snapshot of initial conditions        */
    "out=",                       /* stream of output snapshots            */

    /* params, used if no input specified, to make a Plummer Model         */
    "nbody=16384",                /* number of particles to generate       */
    "seed=123",                   /* random number generator seed          */

    /* params to control N-body integration                                */
    "dtime=0.025",                /* integration time-step                 */
    "eps=0.05",                   /* usual potential softening             */
    "tol=1.0",                    /* cell subdivision tolerence            */
    "fcells=2.0",                 /* cell allocation parameter             */
    "fleaves=0.5",                 /* leaf allocation parameter             */

    "tstop=0.075",                 /* time to stop integration              */
    "dtout=0.25",                 /* data-output interval                  */

    "NPROC=1",                    /* number of processors                  */
  };

  void Help();
  void tab_init();

  int maxleaf;
  int maxcell;
  int maxmybody;
  int maxmycell;
  int maxmyleaf;
  
  // these are used instead of proc. 0's data structures
  bodyptr *mybodytab;
  cellptr *mycelltab;
  leafptr *myleaftab;

  
  public:
  Main(CkArgMsg *m);
  Main(CkMigrateMessage *m);
  void startSimulation();
};

class ParticleChunk : public CBase_ParticleChunk {

  int mynbody;
  int mynumcell;
  int mynumleaf;

  bodyptr *mybodytab;
  cellptr G_root; // obtained from 0th member of array,
                  // after it has finished creating the
                  // top level tree
  nodeptr Current_Root;
  int Root_Coords[NDIM];

  real tnow;
  real tstop;
  int nstep;

  CkCallback mainCb;

  // how many messages containing particles have been sent to 
  // each of the top-level treepieces
  int *numMsgsToEachTp;
  // which of my particles are to be sent to which top-level 
  // treepieces
  CkVec<CkVec<bodyptr> >particlesToTps;

  public:
  ParticleChunk(CkArgMsg *m);
  ParticleChunk(CkMigrateMessage *m);
  void init_root(unsigned int ProcessId);
  void createTopLevelTree(cellptr node, int depth);
  cellptr makecell(unsigned int ProcessId);
  void flushParticles();
  void sendParticlesToTp(int tp);

  void SlaveStart(bodyptr *, cellptr *, leafptr *, CkCallback &cb);
  void startIteration(CkCallback &cb);
  void acceptRoot(cellptr);
  void stepsystemPartII(CkReductionMsg *msg);
  void stepsystemPartIII(CkReductionMsg *msg);

};

class TreePiece : public CBase_TreePiece {
  
  bool isTopLevel;
  int numTotalMsgs;
  int numRecvdMsgs;
  bool haveChildren;
  bool haveCounts;
  int sentTo[NSUB]; // one counter for each child
  CkVec<body> myParticles;
  int myNumParticles;

  nodeptr myRoot;
  nodeptr parent;
  int whichChildAmI;
  int childrenTreePieces[NSUB];

  void checkCompletion();

  public:
  TreePiece(nodeptr parent, int whichChild, bool isTopLevel);
  // used to convey message counts from chunks to top-level
  // treepieces
  void recvTotalMsgCountsFromChunks(CkReductionMsg *msg);
  // used to convey message counts from treepieces to their
  // children
  void recvTotalMsgCountsFromPieces(CkReductionMsg *msg);

};

class ParticleMsg : public CMessage_ParticleMsg {
  public:
  bodyptr particles;
  int num;
};

int subindex(int *xp, int level);
int intcoord(int *xp, vector p);

#endif
