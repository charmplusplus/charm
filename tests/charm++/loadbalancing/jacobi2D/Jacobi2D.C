
#include "Jacobi2D.h"

#define ITERATIONS 10

  TheMain::TheMain(CkArgMsg *)
  {
   CkPrintf("ChareArray %d sq, data array %d sq, iterations %d\n", TheMain::NUM_CHUNKS, TheMain::CHUNK_SIZE, ITERATIONS);
   CProxy_JacobiChunk jc = CProxy_JacobiChunk::ckNew(TheMain::NUM_CHUNKS,TheMain::NUM_CHUNKS);
   jc(0,0).setStartTime(CmiWallTimer());
   jc.startNextIter();
  }
  void TheMain::pup(PUP::er &p) {
   CBase_TheMain::pup(p);
  }

  JacobiChunk::JacobiChunk() {
   int i;
   int j;
   for(i = 1;(i <= TheMain::CHUNK_SIZE);(i++))   for(j = 1;(j <= TheMain::CHUNK_SIZE);(j++))   data[i][j] = (100.0 + (((i + j) % 2)?(-1):1));
   numGot = 0;
   numIters = 0;
   numDone = 0;
   numNeighbors = 4;
   if (((thisIndex.x == 0) || (thisIndex.x == (TheMain::NUM_CHUNKS - 1))))
   (--numNeighbors);
   if (((thisIndex.y == 0) || (thisIndex.y == (TheMain::NUM_CHUNKS - 1))))
   (--numNeighbors);
   maxDelta = 0.0;
   numIters = ITERATIONS;
   usesAtSync = true;
   usesAutoMeasure = true;
  }

  void JacobiChunk::setStartTime(double t)
  {
     CkPrintf("Start time = %f\n", t);
     startTime = t;
  }

  void JacobiChunk::startNextIter()
  {
    int i;
    startT = CmiWallTimer();
    if (thisIndex.x == 0 && thisIndex.y == 0) CmiPrintf("startNextIter: %d\n", numIters);
    if ((thisIndex.x > 0))
    thisProxy((thisIndex.x - 1),thisIndex.y).getBottom(data[1]);
    if ((thisIndex.x < (TheMain::NUM_CHUNKS - 1)))
    thisProxy((thisIndex.x + 1),thisIndex.y).getTop(data[TheMain::CHUNK_SIZE]);
    float tmp[TheMain::CHUNK_SIZE + 2];
    if ((thisIndex.y > 0))
    {
     for(i = 0;(i <= (TheMain::CHUNK_SIZE + 1));(i++))     tmp[i] = data[i][1];
     thisProxy(thisIndex.x,(thisIndex.y - 1)).getRight(tmp);
    }
    if ((thisIndex.y < (TheMain::NUM_CHUNKS - 1)))
    {
     for(i = 0;(i <= (TheMain::CHUNK_SIZE + 1));(i++))     tmp[i] = data[i][TheMain::CHUNK_SIZE];
     thisProxy(thisIndex.x,(thisIndex.y + 1)).getLeft(tmp);
    }
  }

  void JacobiChunk::getLeft(float left[])
  {
   int i;
   for(i = 1;(i <= TheMain::CHUNK_SIZE);(i++))   data[i][0] = left[i];
   if (((++numGot) == numNeighbors))
   {
    numGot = 0;
    refine();
   }
  }
  void JacobiChunk::getRight(float right[])
  {
   int i;
   for(i = 1;(i <= TheMain::CHUNK_SIZE);(i++))   data[i][(TheMain::CHUNK_SIZE + 1)] = right[i];
   if (((++numGot) == numNeighbors))
   {
    numGot = 0;
    refine();
   }
  }
  void JacobiChunk::getTop(float top[])
  {
   int i;
   for(i = 1;(i <= TheMain::CHUNK_SIZE);(i++))   data[0][i] = top[i];
   if (((++numGot) == numNeighbors))
   {
    numGot = 0;
    refine();
   }
  }
  void JacobiChunk::getBottom(float bottom[])
  {
   int i;
   for(i = 1;(i <= TheMain::CHUNK_SIZE);(i++))   data[(TheMain::CHUNK_SIZE + 1)][i] = bottom[i];
   if (((++numGot) == numNeighbors))
   {
    numGot = 0;
    refine();
   }
  }
  void JacobiChunk::refine()
  {
//      double t = CmiWallTimer();
   int i;
   int j;
   if ((thisIndex.y == 0))
   for(i = 1;(i <= TheMain::CHUNK_SIZE);(i++))   data[i][0] = data[i][1];
   else
   if ((thisIndex.y == (TheMain::NUM_CHUNKS - 1)))
   for(i = 1;(i <= TheMain::CHUNK_SIZE);(i++))   data[i][(TheMain::CHUNK_SIZE + 1)] = data[i][TheMain::CHUNK_SIZE];
   if ((thisIndex.x == 0))
   for(i = 0;(i <= TheMain::CHUNK_SIZE);(i++))   data[0][i] = data[1][i];
   else
   if ((thisIndex.x == (TheMain::NUM_CHUNKS - 1)))
   for(i = 0;(i <= TheMain::CHUNK_SIZE);(i++))   data[(TheMain::CHUNK_SIZE + 1)][i] = data[TheMain::CHUNK_SIZE][i];
   for (int k=0; k<(thisIndex.x*TheMain::NUM_CHUNKS + thisIndex.y)*100; k++)
   for(i = 1;(i <= TheMain::CHUNK_SIZE);(i++))   for(j = 1;(j <= TheMain::CHUNK_SIZE);(j++))   data[i][j] = (((((data[(i - 1)][j] + data[i][j]) + data[(i + 1)][j]) + data[i][(j - 1)]) + data[i][(j + 1)]) / 5.0);
//     CkPrintf("Iteration time in microsecs = %d\n", (int )((CmiWallTimer() - t) * 1.0e6));
   if (false)
   {
     int i,j;
     maxDelta = 0;
     for(i = 1;(i <= TheMain::CHUNK_SIZE);(i++))    for(j = 1;(j <= TheMain::CHUNK_SIZE);(j++))    {
      float delta = (data[i][j] - data[(i - 1)][j]);
      if ((delta < 0))
      delta *= (-1);
      if ((delta > maxDelta))
      maxDelta = delta;
     }
   }

   double t = CmiWallTimer() - startT;
   CkCallback cb2(CkIndex_JacobiChunk::print(NULL), thisProxy(0,0));
   contribute(sizeof(double), &t, CkReduction::max_double, cb2);

     // reduction
   CkCallback cb(CkIndex_JacobiChunk::stepping(NULL), thisProxy(0,0));
   contribute(sizeof(float), &maxDelta, CkReduction::max_float, cb);
  }

  void JacobiChunk::done(float delta)
  {
   if ((delta > maxDelta))
     maxDelta = delta;
   double t = CmiWallTimer();
   CkPrintf("From %d %d: %f\n", thisIndex.x, thisIndex.y, maxDelta);
   CkPrintf("End time = %f, Total time in microsecs = %u\n", t, (unsigned long)((t - startTime) * 1.0e6));
   CkExit();
  }
  void JacobiChunk::pup(PUP::er &p) {
   CBase_JacobiChunk::pup(p);
   p|startTime;
   p|numNeighbors;
   p|numDone;
   p|numGot;
   p|numIters;
   p|maxDelta;
   p((char*)data, (TheMain::CHUNK_SIZE+2)*(TheMain::CHUNK_SIZE+2)*sizeof(float));
  }

  void JacobiChunk::ResumeFromSync() {
   startNextIter();
  }

  // a callback function to set object load
  void JacobiChunk::UserSetLBLoad() {
    setObjTime(1.0);
  }

  void JacobiChunk::print(CkReductionMsg *m) {   // on PE 0
   double maxT = *(double*)m->getData();
   CmiPrintf("Iter: %d takes %fs.\n", numIters, maxT);
  }

  void JacobiChunk::stepping(CkReductionMsg *m) {   // on PE 0
   float max_delta = *(float*)m->getData();
   delete m;
   numIters --;
   if (numIters > 0) {
     if (numIters % MIGRATE_STEPS == 5) {
       thisProxy.AtSync();
     }
     else
       thisProxy.startNextIter();
   }
   else
       done(max_delta);
  }


#include "Jacobi2D.def.h"
