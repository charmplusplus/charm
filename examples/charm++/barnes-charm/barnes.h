#ifndef _BARNES_H_
#define _BARNES_H_

#include <string>
#include "defs.h"
#include "stdinc.h"

using namespace std;

#include "barnes.decl.h"

#define MAX_PARTS_PER_TP 1000
#define MAX_PARTICLES_PER_MSG 500

#define G_MALLOC malloc


extern CProxy_Main mainChare;
extern CProxy_TreePiece pieces;
extern CProxy_ParticleChunk chunks;

extern int numTreePieces;
extern int numParticleChunks;
extern int nbody;
extern int NPROC;
extern int maxmybody;
extern real fcells;
extern real fleaves;
extern real tstop;
extern int nbody;
extern real dtime;
extern real eps;
extern real tol;
extern real dtout;
extern real dthf;
extern vector rmin;
extern real rsize;


class ParticleMsg : public CMessage_ParticleMsg {
  public:
  bodyptr *particles;
  int num;
};

/*
const char *defv[] = {                 
    "in=",                        
    "out=",                       

    "nbody=16384",                
    "seed=123",                   

    "dtime=0.025",                
    "eps=0.05",                   
    "tol=1.0",                    
    "fcells=2.0",                 
    "fleaves=0.5",                

    "tstop=0.075",                
    "dtout=0.25",                 

    "NPROC=1"                    
};
*/

static string headline = "Hack code: Plummer model";

class Main : public CBase_Main {

  
  int maxleaf;
  int maxcell;
  int maxmybody;
  int maxmycell;
  int maxmyleaf;

  string infile;
  string outfile;

  real tnow;
  real tout;
  real mymtot;
  
  // these are used instead of proc. 0's data structures
  bodyptr *mybodytab;
  bodyptr bodytab;

  CkVec<string> defaults; 

  // j: don't need these data structures
  //cellptr *mycelltab;
  //leafptr *myleaftab;

  //void initparam (string *argv, const char **defv);
  void initoutput();

  string extrvalue(string &arg);
  //bool matchname(string &bind, string &name);
  //int scanbind(CkVec<string> &bvec, string &name);
  double getdparam(string name);
  bool getbparam(string name);
  long getlparam(string name);
  int getiparam(string name);
  string getparam(string name);

  
  public:
  Main(CkArgMsg *m);
  Main(CkMigrateMessage *m){}
  void startSimulation();
  void pickshell(real vec[], real rad);
  void testdata();
  void startrun();
  void Help();
  void tab_init();
  void setbound();

};

class ParticleChunk : public CBase_ParticleChunk {

  int mynbody;
  // pointer to my pointers to particles
  bodyptr *mybodytab;
  // pointer to array of particles (required for find_my_initial_bodies)
  bodyptr bodytab;

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
  ParticleChunk(int maxleaf, int maxcell);
  ParticleChunk(CkArgMsg *m);
  ParticleChunk(CkMigrateMessage *m){}
  void init_root(unsigned int ProcessId);
  void find_my_initial_bodies(bodyptr btab, int nb, unsigned ProcessId);

  void createTopLevelTree(cellptr node, int depth);
  void flushParticles();
  void sendParticlesToTp(int tp);

  //void SlaveStart(bodyptr *bb, bodyptr b, CkCallback &cb);
  void SlaveStart(CmiUInt8 bb, CmiUInt8 b, CkCallback &cb);
  void startIteration(CkCallback &cb);
  void acceptRoot(CmiUInt8 root);
  void stepsystem(unsigned ProcessId);
  void stepsystemPartII(CkReductionMsg *msg);
  void stepsystemPartIII(CkReductionMsg *msg);

  void maketree(unsigned int ProcessId);
  nodeptr loadtree(bodyptr p, cellptr root, unsigned ProcessId);
  void doneSendingParticles();

  void doneTreeBuild();

};

class TreePiece : public CBase_TreePiece {

  int mynumcell;
  int mynumleaf;
 
  bool isTopLevel;
  int myLevel;
  int numTotalMsgs;
  int numRecvdMsgs;
  bool haveChildren;
  bool haveCounts;
  int sentTo[NSUB]; // one counter for each child
  CkVec<bodyptr> myParticles;
  int myNumParticles;

  cellptr myRoot;
  nodeptr parent;
  int whichChildAmI;
  int childrenTreePieces[NSUB];

  void checkCompletion();

  public:
  TreePiece(nodeptr parent, int whichChild, bool isTopLevel, int level);
  TreePiece(CkMigrateMessage *m){}
  // used to convey message counts from chunks to top-level
  // treepieces
  void recvTotalMsgCountsFromChunks(CkReductionMsg *msg);
  // used to convey message counts from treepieces to their
  // children
  void recvTotalMsgCountsFromPieces(int num);
  void buildTree();
  void recvParticles(ParticleMsg *msg);
  nodeptr loadtree(bodyptr p, cellptr root, unsigned int ProcessId);
  cellptr SubdivideLeaf (leafptr le, cellptr parent, unsigned int l, unsigned int ProcessId);

};

int subindex(int *xp, int level);
bool intcoord(int *xp, vector p);
cellptr makecell(unsigned ProcessId);
leafptr makeleaf(unsigned ProcessId);
leafptr InitLeaf(cellptr parent, unsigned ProcessId);
cellptr InitCell(cellptr parent, unsigned ProcessId);
int log8floor(int arg);

real xrand(real lo, real hi);

#endif
