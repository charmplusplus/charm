
#include <vector>
using std::vector;
#include "pup_stl.h"

#include "Jacobi2D.decl.h"

#define  MIGRATE_STEPS              20

 class TheMain: public CBase_TheMain  {
  public: enum {NUM_CHUNKS=20};
  public: enum {CHUNK_SIZE=64};
  public: TheMain(CkArgMsg *);
  public: virtual void pup(PUP::er &p);
 };

typedef float arr_t[TheMain::CHUNK_SIZE+2][TheMain::CHUNK_SIZE+2];
PUPbytes(arr_t)

 class JacobiChunk: public CBase_JacobiChunk  {
  public: JacobiChunk(CkMigrateMessage *m) {}
  private: float data[TheMain::CHUNK_SIZE+2][TheMain::CHUNK_SIZE+2];
  private: int numIters;
  private: int numGot;
  private: int numDone;
  private: int numNeighbors;
  private: float maxDelta;
  private: double startTime;
  public: JacobiChunk();
  public: void setStartTime(double t);
  public: void startNextIter();
  public: void getLeft(float left[]);
  public: void getRight(float right[]);
  public: void getTop(float top[]);
  public: void getBottom(float bottom[]);
  public: void refine();
  public: void done(float delta);
  public: virtual void pup(PUP::er &p);
  public: virtual void ResumeFromSync();
  public: void stepping(CkReductionMsg *m);
 };

