#include "ckdirect.h" 

#if defined ARR_CHECK && !defined CMK_PARANOID
#define CMK_PARANOID
#endif

#include "stencil3d.decl.h"
#define OOB 9999999999.0
#define NBRS 6

CProxy_Main mainProxy;

int dimx, dimy, dimz;
int charesx, charesy, charesz;
int num_iterations;

class Main : public CBase_Main {
  
  double startSetup, start, end;
  public:

  Main(CkArgMsg *msg){
    mainProxy = thisProxy;

    if(msg->argc == 8){
      dimx = atoi(msg->argv[1]);
      dimy = atoi(msg->argv[2]);
      dimz = atoi(msg->argv[3]);
      charesx = atoi(msg->argv[4]);
      charesy = atoi(msg->argv[5]);
      charesz = atoi(msg->argv[6]);
      num_iterations = atoi(msg->argv[7]);
    }
    else{
      CkAbort("arguments: dimx dimy dimz charesx charesy charesz iter\n");
    }
    
    delete msg;
    CProxy_StencilPoint array = 
        CProxy_StencilPoint::ckNew(charesx*charesy*charesz);
    CkPrintf("main: dim: %d,%d,%d charedim: %d,%d,%d num_iterations: %d\n", dimx, dimy, dimz, charesx, charesy, charesz, num_iterations);
    startSetup = CkWallTimer();
#ifdef USE_CKDIRECT
    array.setupChannels();
#else
    array.sendData();
#endif
  }

  void doneSetup(){
    start = CkWallTimer();
  }
  
  void done(CkReductionMsg *msg){
    end = CkWallTimer();
    CkPrintf("Total time: %f sec\n", end-startSetup);
    CkPrintf("Computation time per iteration: %f sec\n", (end-start)/(num_iterations-1));
    CkPrintf("Total computations: %f\n", *(double *)msg->getData());
    CkExit();
  }
};

class StencilMsg : public CMessage_StencilMsg {
  public:
  float *arr;
  int size;
  int which;
};

class StencilPoint : public CBase_StencilPoint{
  // this chare's place in the array of chares
  int row, col, plane;
  int whichLocal;

  // this chare has a domain with dimensions 
  int rows, cols, planes;
  int payload[3];
  
  int iterations;
  
    /* Number of elements whose new values have been computed
     */
    double eltsComp;
//#ifdef USE_CKDIRECT
  /* 0: left, 1: right, 2: top, 3: bottom*/
  infiDirectUserHandle shandles[2][NBRS];
  infiDirectUserHandle rhandles[NBRS];

//#endif
#ifdef ARR_CHECK
#ifdef USE_MESSAGES
  float *recvBuf[NBRS];
#else
  CkVec<float> recvBuf[NBRS];
#endif
  CkVec<float> sendBuf[2][NBRS];
  CkVec<float> localChunk[2];
#else
  /* send buffers */
  float *sendBuf[2][NBRS]; /* rectangle */
  /* receive buffers */
  float *recvBuf[NBRS]; /* rectangle */
  float *localChunk[2]; /* cuboid */
#endif
#ifdef USE_MESSAGES
  StencilMsg *recvMsgs[NBRS];
#endif

  int remainingBufs;
  int remainingChannels;
  
  public:
  StencilPoint(CkMigrateMessage *){}
  StencilPoint(){
    remainingBufs = NBRS;
    remainingChannels = NBRS;
    whichLocal= 0;

    iterations = 0;
    eltsComp = 0;
    
    plane = thisIndex/(charesx*charesy);
    row = (thisIndex/charesx)%charesy;
    col = thisIndex%charesx;
    
    rows = dimy/charesy;
    cols = dimx/charesx;
    planes = dimz/charesz;

    payload[0] = rows*planes; // x
    payload[1] = cols*planes; // y
    payload[2] = rows*cols;   // z
    
    if(thisIndex == 0){
      CkPrintf("rows: %d cols: %d planes: %d\n", rows, cols, planes);
    }
    // allocate memory
#ifdef ARR_CHECK
    sendBuf[0][0].resize(payload[0]);
    sendBuf[1][0].resize(payload[0]);
    sendBuf[0][1].resize(payload[0]);
    sendBuf[1][1].resize(payload[0]);

    sendBuf[0][2].resize(payload[1]);
    sendBuf[1][2].resize(payload[1]);
    sendBuf[0][3].resize(payload[1]);
    sendBuf[1][3].resize(payload[1]);
    
    sendBuf[0][4].resize(payload[2]);
    sendBuf[1][4].resize(payload[2]);
    sendBuf[0][5].resize(payload[2]);
    sendBuf[1][5].resize(payload[2]);
#else
    sendBuf[0][0] = new float [payload[0]];
    sendBuf[1][0] = new float [payload[0]];
    sendBuf[0][1] = new float [payload[0]];
    sendBuf[1][1] = new float [payload[0]];

    sendBuf[0][2] = new float [payload[1]];
    sendBuf[1][2] = new float [payload[1]];
    sendBuf[0][3] = new float [payload[1]];
    sendBuf[1][3] = new float [payload[1]];

    sendBuf[0][4] = new float [payload[2]];
    sendBuf[1][4] = new float [payload[2]];
    sendBuf[0][5] = new float [payload[2]];
    sendBuf[1][5] = new float [payload[2]];
#endif

#ifndef USE_MESSAGES
#ifdef ARR_CHECK
    recvBuf[0].resize(payload[0]);
    recvBuf[1].resize(payload[0]);
    recvBuf[2].resize(payload[1]);
    recvBuf[3].resize(payload[1]);
    recvBuf[4].resize(payload[2]);
    recvBuf[5].resize(payload[2]);
#else
    recvBuf[0] = new float[payload[0]];
    recvBuf[1] = new float[payload[0]];
    recvBuf[2] = new float[payload[1]];
    recvBuf[3] = new float[payload[1]];
    recvBuf[4] = new float[payload[2]];
    recvBuf[5] = new float[payload[2]];
#endif
#else
    for(int i = 0; i < NBRS; i++)
      recvBuf[i] = 0;
#endif
    
#ifdef ARR_CHECK
    localChunk[0].resize((rows-2)*(cols-2)*(planes-2));
    localChunk[1].resize((rows-2)*(cols-2)*(planes-2));
#else
    localChunk[0] = new float[(rows-2)*(cols-2)*(planes-2)];
    localChunk[1] = new float[(rows-2)*(cols-2)*(planes-2)];
#endif
    // initialize
    if(plane == 0){// face is held at 0
      for(int i = 0; i < rows*cols; i++)
        // first index says which version of the double buffer is to be used
        // second index, the face in question
        sendBuf[0][5][i] = 1.0;
    }
    else{
      for(int i = 0; i < rows*cols; i++)
        // first index says which version of the double buffer is to be used
        // second index, the face in question
        sendBuf[0][5][i] = 0.0;
    }

    for(int i = 0; i < rows*cols; i++)
      sendBuf[0][4][i] = 0.0;
      
    for(int i = 0; i < rows*planes; i++)
      sendBuf[0][1][i] = 0.0;
      
    for(int i = 0; i < rows*planes; i++)
      sendBuf[0][0][i] = 0.0;
      
    for(int i = 0; i < cols*planes; i++)
      sendBuf[0][3][i] = 0.0;
      
    for(int i = 0; i < cols*planes; i++)
      sendBuf[0][2][i] = 0.0;
      
    
    // now the rest of the cube
    for(int j = 0; j < (rows-2)*(cols-2)*(planes-2); j++){
      localChunk[0][j] = 0.0; 
    }
  }
  
  ~StencilPoint(){
    for(int i = 0; i < NBRS; i++){
#ifdef ARR_CHECK
      sendBuf[0][i].free();
      sendBuf[1][i].free();
#else
      delete [] sendBuf[0][i];
      delete [] sendBuf[1][i];
#endif
#ifndef USE_MESSAGES
#ifdef ARR_CHECK
      recvBuf[i].free();
#else
      delete [] recvBuf[i];
#endif
#endif
    }

#ifdef ARR_CHECK
    localChunk[0].free();
    localChunk[1].free();
#else
    delete [] localChunk[0];
    delete [] localChunk[1];
#endif
  }

#define lin(c,r,p) ((p)*charesx*charesy+(r)*charesx+(c))
//#ifdef USE_CKDIRECT
  void setupChannels(){
    int node = CkMyNode();
#ifdef STENCIL2D_VERBOSE
    /*
    CkPrintf("(%d,%d): notify (%d,%d):l, (%d,%d):r, (%d,%d):t, (%d,%d):b\n", 
              row, col,
              row, (col+1)%num_cols,
              row, (col-1+num_cols)%num_cols,
              (row+1)%num_rows, col,
              (row-1+num_rows)%num_rows, col);
    */
#endif

    thisProxy[lin((col+1)%charesx, (row), (plane))].notify(node,0,thisIndex);
    thisProxy[lin((col-1+charesx)%charesx, (row), (plane))].notify(node,1,thisIndex);
    thisProxy[lin((col), (row+1)%charesy, (plane))].notify(node,2,thisIndex);
    thisProxy[lin((col), (row-1+charesy)%charesy, (plane))].notify(node,3,thisIndex);
    thisProxy[lin((col), (row), (plane+1)%charesz)].notify(node,4,thisIndex);
    thisProxy[lin((col), (row), (plane-1+charesz)%charesz)].notify(node,5,thisIndex);
  }

  void notify(int node, int which, CkIndex1D whoSent){
    // make handle
#ifdef STENCIL2D_VERBOSE
    CkPrintf("(%d,%d,%d): (%d) is %d to me\n", row, col, plane, whoSent, which);
#endif
#if defined ARR_CHECK && !defined USE_MESSAGES
    rhandles[which] = CkDirect_createHandle(node, recvBuf[which].getVec(), payload[which/2]*sizeof(float), StencilPoint::callbackWrapper, (void *)this, OOB);
#else
    rhandles[which] = CkDirect_createHandle(node, recvBuf[which], payload[which/2]*sizeof(float), StencilPoint::callbackWrapper, (void *)this, OOB);
#endif
    thisProxy[whoSent].recvHandle(rhandles[which], which);
  }

  void recvHandle(infiDirectUserHandle handle, int which){
    // this causes an inversion of the buffer indices:
    // left send buffer is 1, right 0, top 3, bottom 2
    shandles[0][which] = handle;
    shandles[1][which] = handle;
    
#ifdef ARR_CHECK
    CkDirect_assocLocalBuffer(&shandles[0][which], sendBuf[0][which].getVec(), payload[which/2]*sizeof(float));
    CkDirect_assocLocalBuffer(&shandles[1][which], sendBuf[1][which].getVec(), payload[which/2]*sizeof(float));
#else
    CkDirect_assocLocalBuffer(&shandles[0][which], sendBuf[0][which], payload[which/2]*sizeof(float));
    CkDirect_assocLocalBuffer(&shandles[1][which], sendBuf[1][which], payload[which/2]*sizeof(float));
#endif
    remainingChannels--;
    if(remainingChannels == 0){
      // start 
#ifdef STENCIL2D_VERBOSE
      //CkPrintf("(%d,%d,%d): recvd all handles, start put\n", x,y,z);
#endif
      sendData();
    }
  }

  void recvBuffer(){
    remainingBufs--;
#ifdef STENCIL2D_VERBOSE
    //CkPrintf("(%d,%d,%d): remainingBufs: %d\n", x,y,z,remainingBufs);
#endif
    if(remainingBufs == 0){
#ifdef STENCIL2D_VERBOSE
      //CkPrintf("(%d,%d,%d): recvd all buffers, start compute(%d)\n", x,y,z, iterations);
#endif
      remainingBufs = NBRS;
      thisProxy[thisIndex].compute();
    }
  }
  
  static void callbackWrapper(void *arg){
    StencilPoint *obj = (StencilPoint *)arg;
    obj->recvBuffer();
  }
//#else   // don't USE_CKDIRECT
  void recvBuffer(float *array, int size, int whichBuf){
    remainingBufs--;
#if defined ARR_CHECK && !defined USE_MESSAGES
    memcpy(recvBuf[whichBuf].getVec(), array, size*sizeof(float));
#else
    memcpy(recvBuf[whichBuf], array, size*sizeof(float));
#endif
    /*
    for(int i = 0; i < size; i++){
      recvBuf[whichBuf][i] = array[i];
    }
    */
    
    if(remainingBufs == 0){
      remainingBufs = NBRS;
      compute();
    }
  }

  void recvBufferMsg(StencilMsg *msg){
#ifdef USE_MESSAGES
    remainingBufs--;
    recvBuf[msg->which] = msg->arr;
    recvMsgs[msg->which] = msg;
    
    if(remainingBufs == 0){
      remainingBufs = NBRS;
      compute();
      for(int i = 0; i < NBRS; i++){
        delete recvMsgs[i];
        recvMsgs[i] = 0;
      }
    }
 
#else
      CkAbort("Messages not used, don't call\n");
#endif
      
  }
//#endif // end ifdef USE_CKDIRECT
  

// to access localChunk interior (i.e. interior of interior) 
#define small(r,c,p) (p)*(rows-2)*(cols-2)+(r)*(cols-2)+(c)
// to access sendBufs 
#define facex(r,c) (r)*planes+(c)
#define facey(r,c) (r)*planes+(c)
#define facez(r,c) (r)*cols+(c)
// indices into sendBufs, after inversion
#define xp 1
#define xn 0
#define yp 3
#define yn 2
#define zp 5
#define zn 4

// indices into recvBufs, without inversion
#define XP 0
#define XN 1
#define YP 2
#define YN 3
#define ZP 4
#define ZN 5

  void compute(){
    // 1. do actual work
    int newLocal= 1-whichLocal;
    
    // interior of interior: uses only localChunk
    for(int k = 1; k < planes-3; k++){
      for(int i = 1; i < rows-3; i++){
        for(int j = 1; j < cols-3; j++){
          localChunk[newLocal][small(i,j,k)] = (
                              localChunk[whichLocal][small(i,j,k)]+
                              localChunk[whichLocal][small(i+1,j,k)]+
                              localChunk[whichLocal][small(i-1,j,k)]+
                              localChunk[whichLocal][small(i,j+1,k)]+
                              localChunk[whichLocal][small(i,j-1,k)]+
                              localChunk[whichLocal][small(i,j,k+1)]+
                              localChunk[whichLocal][small(i,j,k-1)]
                            )/7;
          eltsComp+=1;
        }
      }
    }

    // 8 corners of interior: uses localChunk and sendBuf's
    //0.
    localChunk[newLocal][small(0,0,0)] = (
                            localChunk[whichLocal][small(0,0,0)]+
                            localChunk[whichLocal][small(1,0,0)]+
                            localChunk[whichLocal][small(0,1,0)]+
                            localChunk[whichLocal][small(0,0,1)]+
                            sendBuf[whichLocal][zp][facez(1,1)]+
                            sendBuf[whichLocal][xp][facex(1,1)]+
                            sendBuf[whichLocal][yp][facey(1,1)]
                            )/7;

    //1.
    localChunk[newLocal][small(0,cols-3,0)] = (
                            localChunk[whichLocal][small(0,cols-3,0)]+
                            localChunk[whichLocal][small(0,cols-4,0)]+
                            localChunk[whichLocal][small(1,cols-3,0)]+
                            localChunk[whichLocal][small(0,cols-3,1)]+
                            sendBuf[whichLocal][zp][facez(1,cols-2)]+
                            sendBuf[whichLocal][yp][facey(cols-2,1)]+
                            sendBuf[whichLocal][xn][facex(1,1)]
                            )/7;

    //2.
    localChunk[newLocal][small(rows-3,0,0)] = (
                            localChunk[whichLocal][small(rows-3,0,0)]+
                            localChunk[whichLocal][small(rows-4,0,0)]+
                            localChunk[whichLocal][small(rows-3,1,0)]+
                            localChunk[whichLocal][small(rows-3,0,1)]+
                            sendBuf[whichLocal][zp][facez(rows-2,1)]+
                            sendBuf[whichLocal][yn][facey(1,1)]+
                            sendBuf[whichLocal][xp][facex(rows-2,1)]
                            )/7;

    //3.
    localChunk[newLocal][small(rows-3,cols-3,0)] = (
                            localChunk[whichLocal][small(rows-3,cols-3,0)]+
                            localChunk[whichLocal][small(rows-3,cols-4,0)]+
                            localChunk[whichLocal][small(rows-4,cols-3,0)]+
                            localChunk[whichLocal][small(rows-3,cols-3,1)]+
                            sendBuf[whichLocal][zp][facez(rows-2,cols-2)]+
                            sendBuf[whichLocal][yn][facey(cols-2,1)]+
                            sendBuf[whichLocal][xn][facex(rows-2,1)]
                            )/7;
    
    //4.
    localChunk[newLocal][small(0,0,planes-3)] = (
                            localChunk[whichLocal][small(0,0,planes-3)]+
                            localChunk[whichLocal][small(0,0,planes-4)]+
                            localChunk[whichLocal][small(0,1,planes-3)]+
                            localChunk[whichLocal][small(1,0,planes-3)]+
                            sendBuf[whichLocal][zn][facez(1,1)]+
                            sendBuf[whichLocal][yp][facey(1,planes-2)]+
                            sendBuf[whichLocal][xp][facex(1,planes-2)]
                            )/7;

    //5.
    localChunk[newLocal][small(0,cols-3,planes-3)] = (
                            localChunk[whichLocal][small(0,cols-3,planes-3)]+
                            localChunk[whichLocal][small(0,cols-4,planes-3)]+
                            localChunk[whichLocal][small(1,cols-3,planes-3)]+
                            localChunk[whichLocal][small(0,cols-3,planes-4)]+
                            sendBuf[whichLocal][zn][facez(1,cols-2)]+
                            sendBuf[whichLocal][yp][facey(cols-2,planes-2)]+
                            sendBuf[whichLocal][xn][facex(1,planes-2)]
                            )/7;
    
    //6.
    localChunk[newLocal][small(rows-3,0,planes-3)] = (
                            localChunk[whichLocal][small(rows-3,0,planes-3)]+
                            localChunk[whichLocal][small(rows-4,0,planes-3)]+
                            localChunk[whichLocal][small(rows-3,1,planes-3)]+
                            localChunk[whichLocal][small(rows-3,0,planes-4)]+
                            sendBuf[whichLocal][zn][facez(rows-2,1)]+
                            sendBuf[whichLocal][yn][facey(1,planes-2)]+
                            sendBuf[whichLocal][xp][facex(rows-2,planes-2)]
                            )/7;

    //7.
    localChunk[newLocal][small(rows-3,cols-3,planes-3)] = (
                            localChunk[whichLocal][small(rows-3,cols-3,planes-3)]+
                            localChunk[whichLocal][small(rows-4,cols-3,planes-3)]+
                            localChunk[whichLocal][small(rows-3,cols-4,planes-3)]+
                            localChunk[whichLocal][small(rows-3,cols-3,planes-4)]+
                            sendBuf[whichLocal][zn][facez(rows-2,cols-2)]+
                            sendBuf[whichLocal][yn][facey(cols-2,planes-2)]+
                            sendBuf[whichLocal][xn][facex(rows-2,planes-2)]
                            )/7;

   eltsComp += 8;

    // 12 edges
    //zp,yp
    for(int i = 1; i < cols-3; i++){
      localChunk[newLocal][small(0,i,0)] = (
                localChunk[whichLocal][small(0,i,0)]+
                localChunk[whichLocal][small(0,i-1,0)]+
                localChunk[whichLocal][small(0,i+1,0)]+
                localChunk[whichLocal][small(1,i,0)]+
                localChunk[whichLocal][small(0,i,1)]+
                sendBuf[whichLocal][yp][facey(i+1,1)]+
                sendBuf[whichLocal][zp][facez(1,i+1)]
          )/7;
       eltsComp+=1;
    }

    //xn,yp
    for(int i = 1; i < planes-3; i++){
      localChunk[newLocal][small(0,cols-3,i)] = (
                localChunk[whichLocal][small(0,cols-3,i)]+
                localChunk[whichLocal][small(0,cols-3,i-1)]+
                localChunk[whichLocal][small(0,cols-3,i+1)]+
                localChunk[whichLocal][small(1,cols-3,i)]+
                localChunk[whichLocal][small(0,cols-4,i)]+
                sendBuf[whichLocal][xn][facex(1,i+1)]+
                sendBuf[whichLocal][yp][facey(cols-2,i+1)]
          )/7;
       eltsComp+=1;
    }

    //zn,yp
    for(int i = 1; i < cols-3; i++){
      localChunk[newLocal][small(0,i,planes-3)] = (
                localChunk[whichLocal][small(0,i,planes-3)]+
                localChunk[whichLocal][small(0,i-1,planes-3)]+
                localChunk[whichLocal][small(0,i+1,planes-3)]+
                localChunk[whichLocal][small(0,i,planes-4)]+
                localChunk[whichLocal][small(1,i,planes-3)]+
                sendBuf[whichLocal][zn][facez(1,i+1)]+
                sendBuf[whichLocal][yp][facey(i+1,planes-2)]
          )/7;
       eltsComp+=1;
    }

    //yp,xp
    for(int i = 1; i < planes-3; i++){
      localChunk[newLocal][small(0,0,i)] = (
                localChunk[whichLocal][small(0,0,i)]+
                localChunk[whichLocal][small(0,0,i-1)]+
                localChunk[whichLocal][small(0,0,i+1)]+
                localChunk[whichLocal][small(0,1,i)]+
                localChunk[whichLocal][small(1,0,i)]+
                sendBuf[whichLocal][yp][facey(1,i+1)]+
                sendBuf[whichLocal][xp][facex(1,i+1)]
          )/7;
       eltsComp+=1;
    }

    //yn,zp
    for(int i = 1; i < cols-3; i++){
      localChunk[newLocal][small(rows-3,i,0)] = (
                localChunk[whichLocal][small(rows-3,i,0)]+
                localChunk[whichLocal][small(rows-3,i-1,0)]+
                localChunk[whichLocal][small(rows-3,i+1,0)]+
                localChunk[whichLocal][small(rows-4,i,0)]+
                localChunk[whichLocal][small(rows-3,i,1)]+
                sendBuf[whichLocal][yn][facey(i+1,1)]+
                sendBuf[whichLocal][zp][facez(rows-2,i+1)]
          )/7;
       eltsComp+=1;
    }

    //xn,yn
    for(int i = 1; i < planes-3; i++){
      localChunk[newLocal][small(rows-3,cols-3,i)] = (
                localChunk[whichLocal][small(rows-3,cols-3,i)]+
                localChunk[whichLocal][small(rows-3,cols-3,i-1)]+
                localChunk[whichLocal][small(rows-3,cols-3,i+1)]+
                localChunk[whichLocal][small(rows-4,cols-3,i)]+
                localChunk[whichLocal][small(rows-3,cols-4,i)]+
                sendBuf[whichLocal][xn][facex(rows-2,i+1)]+
                sendBuf[whichLocal][yn][facey(cols-2,i+1)]
          )/7;
       eltsComp+=1;
    }

    //yn,zn
    for(int i = 1; i < cols-3; i++){
      localChunk[newLocal][small(rows-3,i,planes-3)] = (
                localChunk[whichLocal][small(rows-3,i,planes-3)]+
                localChunk[whichLocal][small(rows-3,i-1,planes-3)]+
                localChunk[whichLocal][small(rows-3,i+1,planes-3)]+
                localChunk[whichLocal][small(rows-4,i,planes-3)]+
                localChunk[whichLocal][small(rows-3,i,planes-4)]+
                sendBuf[whichLocal][yn][facey(i+1,planes-2)]+
                sendBuf[whichLocal][zn][facez(rows-2,i+1)]
          )/7;
       eltsComp+=1;
    }

    //xp,yn
    for(int i = 1; i < planes-3; i++){
      localChunk[newLocal][small(rows-3,0,i)] = (
                localChunk[whichLocal][small(rows-3,0,i)]+
                localChunk[whichLocal][small(rows-3,0,i-1)]+
                localChunk[whichLocal][small(rows-3,0,i+1)]+
                localChunk[whichLocal][small(rows-3,1,i)]+
                localChunk[whichLocal][small(rows-4,0,i)]+
                sendBuf[whichLocal][xp][facex(rows-2,i+1)]+
                sendBuf[whichLocal][yn][facey(1,i+1)]
          )/7;
       eltsComp+=1;
    }

    //xp,zp
    for(int i = 1; i < rows-3; i++){
      localChunk[newLocal][small(i,0,0)] = (
                localChunk[whichLocal][small(i,0,0)]+
                localChunk[whichLocal][small(i-1,0,0)]+
                localChunk[whichLocal][small(i+1,0,0)]+
                localChunk[whichLocal][small(i,1,0)]+
                localChunk[whichLocal][small(i,0,1)]+
                sendBuf[whichLocal][xp][facex(i+1,1)]+
                sendBuf[whichLocal][zp][facez(i+1,1)]
          )/7;
       eltsComp+=1;
    }

    //xn,zp
    for(int i = 1; i < rows-3; i++){
      localChunk[newLocal][small(i,cols-3,0)] = (
                localChunk[whichLocal][small(i,cols-3,0)]+
                localChunk[whichLocal][small(i-1,cols-3,0)]+
                localChunk[whichLocal][small(i+1,cols-3,0)]+
                localChunk[whichLocal][small(i,cols-4,0)]+
                localChunk[whichLocal][small(i,cols-3,1)]+
                sendBuf[whichLocal][xn][facex(i+1,1)]+
                sendBuf[whichLocal][zp][facez(i+1,cols-2)]
          )/7;
       eltsComp+=1;
    }

    //xp,zn
    for(int i = 1; i < rows-3; i++){
      localChunk[newLocal][small(i,0,planes-3)] = (
                localChunk[whichLocal][small(i,0,planes-3)]+
                localChunk[whichLocal][small(i-1,0,planes-3)]+
                localChunk[whichLocal][small(i+1,0,planes-3)]+
                localChunk[whichLocal][small(i,1,planes-3)]+
                localChunk[whichLocal][small(i,0,planes-4)]+
                sendBuf[whichLocal][xp][facex(i+1,planes-2)]+
                sendBuf[whichLocal][zn][facez(i+1,1)]
          )/7;
       eltsComp+=1;
    }

    //xn,zn
    for(int i = 1; i < rows-3; i++){
      localChunk[newLocal][small(i,cols-3,planes-3)] = (
                localChunk[whichLocal][small(i,cols-3,planes-3)]+
                localChunk[whichLocal][small(i-1,cols-3,planes-3)]+
                localChunk[whichLocal][small(i+1,cols-3,planes-3)]+
                localChunk[whichLocal][small(i,cols-4,planes-3)]+
                localChunk[whichLocal][small(i,cols-3,planes-4)]+
                sendBuf[whichLocal][xn][facex(i+1,planes-2)]+
                sendBuf[whichLocal][zn][facez(i+1,cols-2)]
          )/7;
       eltsComp+=1;

    }

    // 6 more faces - use 6 (including self) from localChunk and 1 from one of the sendBufs
    for(int i = 1; i < rows-3; i++){
      for(int j = 1; j < cols-3; j++){
        localChunk[newLocal][small(i,j,0)] = (
            localChunk[whichLocal][small(i,j,0)]+
            localChunk[whichLocal][small(i-1,j,0)]+
            localChunk[whichLocal][small(i+1,j,0)]+
            localChunk[whichLocal][small(i,j-1,0)]+
            localChunk[whichLocal][small(i,j+1,0)]+
            localChunk[whichLocal][small(i,j,1)]+
            sendBuf[whichLocal][zp][facez(i+1,j+1)]
        )/7;
       eltsComp+=1;
      }
    }
    for(int i = 1; i < rows-3; i++){
      for(int j = 1; j < cols-3; j++){
        localChunk[newLocal][small(i,j,planes-3)] = (
            localChunk[whichLocal][small(i,j,planes-3)]+
            localChunk[whichLocal][small(i-1,j,planes-3)]+
            localChunk[whichLocal][small(i+1,j,planes-3)]+
            localChunk[whichLocal][small(i,j-1,planes-3)]+
            localChunk[whichLocal][small(i,j+1,planes-3)]+
            localChunk[whichLocal][small(i,j,planes-4)]+
            sendBuf[whichLocal][zn][facez(i+1,j+1)]
        )/7;
       eltsComp+=1;
      }
    }
    
    for(int j = 1; j < planes-3; j++){
      for(int i = 1; i < rows-3; i++){
        localChunk[newLocal][small(i,0,j)] = (
            localChunk[whichLocal][small(i,0,j)]+
            localChunk[whichLocal][small(i-1,0,j)]+
            localChunk[whichLocal][small(i+1,0,j)]+
            localChunk[whichLocal][small(i,0,j-1)]+
            localChunk[whichLocal][small(i,0,j+1)]+
            localChunk[whichLocal][small(i,1,j)]+
            sendBuf[whichLocal][xp][facex(i+1,j+1)]
        )/7;
       eltsComp+=1;
      }
    }
    for(int j = 1; j < planes-3; j++){
      for(int i = 1; i < rows-3; i++){
        localChunk[newLocal][small(i,cols-3,j)] = (
            localChunk[whichLocal][small(i,cols-3,j)]+
            localChunk[whichLocal][small(i-1,cols-3,j)]+
            localChunk[whichLocal][small(i+1,cols-3,j)]+
            localChunk[whichLocal][small(i,cols-3,j-1)]+
            localChunk[whichLocal][small(i,cols-3,j+1)]+
            localChunk[whichLocal][small(i,cols-4,j)]+
            sendBuf[whichLocal][xn][facex(i+1,j+1)]
        )/7;
       eltsComp+=1;
      }
    }
    
    for(int j = 1; j < planes-3; j++){
      for(int i = 1; i < cols-3; i++){
        localChunk[newLocal][small(0,i,j)] = (
            localChunk[whichLocal][small(0,i,j)]+
            localChunk[whichLocal][small(0,i-1,j)]+
            localChunk[whichLocal][small(0,i+1,j)]+
            localChunk[whichLocal][small(0,i,j-1)]+
            localChunk[whichLocal][small(0,i,j+1)]+
            localChunk[whichLocal][small(1,i,j)]+
            sendBuf[whichLocal][yp][facey(i+1,j+1)]
        )/7;
       eltsComp+=1;
      }
    }
    for(int j = 1; j < planes-3; j++){
      for(int i = 1; i < cols-3; i++){
        localChunk[newLocal][small(rows-3,i,j)] = (
            localChunk[whichLocal][small(rows-3,i,j)]+
            localChunk[whichLocal][small(rows-3,i-1,j)]+
            localChunk[whichLocal][small(rows-3,i+1,j)]+
            localChunk[whichLocal][small(rows-3,i,j-1)]+
            localChunk[whichLocal][small(rows-3,i,j+1)]+
            localChunk[whichLocal][small(rows-4,i,j)]+
            sendBuf[whichLocal][yn][facey(i+1,j+1)]
        )/7;
       eltsComp+=1;
      }
    }
    /* Now we can compute the sendBufs. there are 6 faces.
     Each element of sendbuf uses the ghost layer (recvbufs) in some way.
     There are elements that :
     a) form the interior of the faces. these use only one ghost element each
     b) are the corners of the faces, that use 3 ghost elements each
     c) are part of the edges, using 2 ghost elements each
    */

    // 1. zp face
    // Interior points first
    // enforce boundary conditions if plane = 0
    if(plane == 0){
      for(int i = 0; i < rows*cols; i++){
        sendBuf[newLocal][zp][i] = 1.0;
        eltsComp+=1;
      }
    }
    else{
    for(int i = 1; i < rows-1; i++){
      for(int j = 1; j < cols-1; j++){
        sendBuf[newLocal][zp][facez(i,j)] = (
                sendBuf[whichLocal][zp][facez(i,j)]+
                sendBuf[whichLocal][zp][facez(i-1,j)]+
                sendBuf[whichLocal][zp][facez(i+1,j)]+
                sendBuf[whichLocal][zp][facez(i,j-1)]+
                sendBuf[whichLocal][zp][facez(i,j+1)]+
                localChunk[whichLocal][small(i-1,j-1,0)]+
                recvBuf[ZP][facez(i,j)]
        )/7;
       eltsComp+=1;
      }
    }

    // Corners next
    // Each element uses 3 ghosts, 3 from its own sendBuf (including itself)  and 1 from a different sendBuf
    sendBuf[newLocal][zp][facez(0,0)] = (
          recvBuf[YP][facey(0,0)]+
          recvBuf[XP][facex(0,0)]+
          recvBuf[ZP][facez(0,0)]+
          sendBuf[whichLocal][zp][facez(0,0)]+
          sendBuf[whichLocal][zp][facez(0,1)]+
          sendBuf[whichLocal][zp][facez(1,0)]+
          sendBuf[whichLocal][xp][facex(0,1)]
    )/7;

    sendBuf[newLocal][zp][facez(0,cols-1)] = (
          recvBuf[YP][facey(cols-1,0)]+
          recvBuf[XN][facex(0,0)]+
          recvBuf[ZP][facez(0,cols-1)]+
          sendBuf[whichLocal][zp][facez(0,cols-1)]+
          sendBuf[whichLocal][zp][facez(1,cols-1)]+
          sendBuf[whichLocal][zp][facez(0,cols-2)]+
          sendBuf[whichLocal][xn][facex(0,1)]
    )/7;

    sendBuf[newLocal][zp][facez(rows-1,0)] = (
          recvBuf[ZP][facez(rows-1,0)]+
          recvBuf[XP][facex(rows-1,0)]+
          recvBuf[YN][facey(0,0)]+
          sendBuf[whichLocal][zp][facez(rows-1,0)]+
          sendBuf[whichLocal][zp][facez(rows-2,0)]+
          sendBuf[whichLocal][zp][facez(rows-1,1)]+
          sendBuf[whichLocal][xp][facex(rows-1,1)]
    )/7;

    sendBuf[newLocal][zp][facez(rows-1,cols-1)] = (
          recvBuf[XN][facex(rows-1,0)]+
          recvBuf[YN][facey(cols-1,0)]+
          recvBuf[ZP][facez(rows-1,cols-1)]+
          sendBuf[whichLocal][zp][facez(rows-1,cols-1)]+
          sendBuf[whichLocal][zp][facez(rows-2,cols-1)]+
          sendBuf[whichLocal][zp][facez(rows-1,cols-2)]+
          sendBuf[whichLocal][xn][facex(rows-1,1)]
    )/7;

    eltsComp += 4;
    // Finally, the edges: 
    // Each element here uses 2 ghosts, 4 elts from its own sendBuf, 1 elt from a different sendBuf
    
    for(int i = 1; i < cols-1; i++){
      sendBuf[newLocal][zp][facez(0,i)] = (
          recvBuf[ZP][facez(0,i)]+
          recvBuf[YP][facey(i,0)]+
          sendBuf[whichLocal][zp][facez(0,i)]+
          sendBuf[whichLocal][zp][facez(0,i-1)]+
          sendBuf[whichLocal][zp][facez(0,i+1)]+
          sendBuf[whichLocal][zp][facez(1,i)]+
          sendBuf[whichLocal][yp][facey(i,1)]
      )/7; 
      eltsComp+=1;
    }
    for(int i = 1; i < rows-1; i++){
      sendBuf[newLocal][zp][facez(i,0)] = (
          recvBuf[XP][facex(i,0)]+
          recvBuf[ZP][facez(i,0)]+
          sendBuf[whichLocal][zp][facez(i,0)]+
          sendBuf[whichLocal][zp][facez(i-1,0)]+
          sendBuf[whichLocal][zp][facez(i+1,0)]+
          sendBuf[whichLocal][zp][facez(i,1)]+
          sendBuf[whichLocal][xp][facex(i,1)]
      )/7;
      eltsComp+=1;
    }
    for(int i = 1; i < cols-1; i++){
      sendBuf[newLocal][zp][facez(rows-1,i)] = (
          recvBuf[YN][facey(i,0)]+
          recvBuf[ZP][facez(rows-1,i)]+
          sendBuf[whichLocal][zp][facez(rows-1,i)]+
          sendBuf[whichLocal][zp][facez(rows-1,i-1)]+
          sendBuf[whichLocal][zp][facez(rows-1,i+1)]+
          sendBuf[whichLocal][zp][facez(rows-2,i)]+
          sendBuf[whichLocal][yn][facey(i,1)]
      )/7;
      eltsComp+=1;
    }
    for(int i = 1; i < rows-1; i++){
      sendBuf[newLocal][zp][facez(i,cols-1)] = (
          recvBuf[XN][facex(i,0)]+
          recvBuf[ZP][facez(i,cols-1)]+
          sendBuf[whichLocal][zp][facez(i,cols-1)]+
          sendBuf[whichLocal][zp][facez(i-1,cols-1)]+
          sendBuf[whichLocal][zp][facez(i+1,cols-1)]+
          sendBuf[whichLocal][zp][facez(i,cols-2)]+
          sendBuf[whichLocal][xn][facex(i,1)]
      )/7;
      eltsComp+=1;
    }
    }
    
    // 2. zn face
    // Interior points first
    for(int i = 1; i < rows-1; i++){
      for(int j = 1; j < cols-1; j++){
        sendBuf[newLocal][zn][facez(i,j)] = (
                sendBuf[whichLocal][zn][facez(i,j)]+
                sendBuf[whichLocal][zn][facez(i-1,j)]+
                sendBuf[whichLocal][zn][facez(i+1,j)]+
                sendBuf[whichLocal][zn][facez(i,j-1)]+
                sendBuf[whichLocal][zn][facez(i,j+1)]+
                localChunk[whichLocal][small(i-1,j-1,planes-3)]+
                recvBuf[ZN][facez(i,j)]
        )/7;
      eltsComp+=1;
      }
    }

    // Corners next
    // Each element uses 3 ghosts, 3 from its own sendBuf (including itself)  and 1 from a different sendBuf
    sendBuf[newLocal][zn][facez(0,0)] = (
          recvBuf[YP][facey(0,planes-1)]+
          recvBuf[XP][facex(0,planes-1)]+
          recvBuf[ZN][facez(0,0)]+
          sendBuf[whichLocal][zn][facez(0,0)]+
          sendBuf[whichLocal][zn][facez(0,1)]+
          sendBuf[whichLocal][zn][facez(1,0)]+
          sendBuf[whichLocal][xp][facex(0,planes-2)]
    )/7;

    sendBuf[newLocal][zn][facez(0,cols-1)] = (
          recvBuf[YP][facey(cols-1,planes-1)]+
          recvBuf[XN][facex(0,planes-1)]+
          recvBuf[ZN][facez(0,cols-1)]+
          sendBuf[whichLocal][zn][facez(0,cols-1)]+
          sendBuf[whichLocal][zn][facez(1,cols-1)]+
          sendBuf[whichLocal][zn][facez(0,cols-2)]+
          sendBuf[whichLocal][xn][facex(0,cols-2)]
    )/7;

    sendBuf[newLocal][zn][facez(rows-1,0)] = (
          recvBuf[ZN][facez(rows-1,0)]+
          recvBuf[XP][facex(rows-1,planes-1)]+
          recvBuf[YN][facey(0,planes-1)]+
          sendBuf[whichLocal][zn][facez(rows-1,0)]+
          sendBuf[whichLocal][zn][facez(rows-2,0)]+
          sendBuf[whichLocal][zn][facez(rows-1,1)]+
          sendBuf[whichLocal][xp][facex(rows-1,planes-2)]
    )/7;

    sendBuf[newLocal][zn][facez(rows-1,cols-1)] = (
          recvBuf[XN][facex(rows-1,planes-1)]+
          recvBuf[YN][facey(cols-1,planes-1)]+
          recvBuf[ZN][facez(rows-1,cols-1)]+
          sendBuf[whichLocal][zn][facez(rows-1,cols-1)]+
          sendBuf[whichLocal][zn][facez(rows-2,cols-1)]+
          sendBuf[whichLocal][zn][facez(rows-1,cols-2)]+
          sendBuf[whichLocal][xn][facex(rows-1,planes-2)]
    )/7;

      eltsComp+=4;
    // Finally, the edges: 
    // Each element here uses 2 ghosts, 4 elts from its own sendBuf, 1 elt from a different sendBuf
    
    for(int i = 1; i < cols-1; i++){
      sendBuf[newLocal][zn][facez(0,i)] = (
          recvBuf[ZN][facez(0,i)]+
          recvBuf[YP][facey(i,planes-1)]+
          sendBuf[whichLocal][zn][facez(0,i)]+
          sendBuf[whichLocal][zn][facez(0,i-1)]+
          sendBuf[whichLocal][zn][facez(0,i+1)]+
          sendBuf[whichLocal][zn][facez(1,i)]+
          sendBuf[whichLocal][yp][facey(i,planes-2)]
      )/7;
      eltsComp+=1;
    }
    for(int i = 1; i < rows-1; i++){
      sendBuf[newLocal][zn][facez(i,0)] = (
          recvBuf[XP][facex(i,planes-1)]+
          recvBuf[ZN][facez(i,0)]+
          sendBuf[whichLocal][zn][facez(i,0)]+
          sendBuf[whichLocal][zn][facez(i-1,0)]+
          sendBuf[whichLocal][zn][facez(i+1,0)]+
          sendBuf[whichLocal][zn][facez(i,1)]+
          sendBuf[whichLocal][xp][facex(i,planes-2)]
      )/7;
      eltsComp+=1;
    }
    for(int i = 1; i < cols-1; i++){
      sendBuf[newLocal][zn][facez(rows-1,i)] = (
          recvBuf[YN][facey(i,planes-1)]+
          recvBuf[ZN][facez(rows-1,i)]+
          sendBuf[whichLocal][zn][facez(rows-1,i)]+
          sendBuf[whichLocal][zn][facez(rows-1,i-1)]+
          sendBuf[whichLocal][zn][facez(rows-1,i+1)]+
          sendBuf[whichLocal][zn][facez(rows-2,i)]+
          sendBuf[whichLocal][yn][facey(i,planes-2)]
      )/7;
      eltsComp+=1;
    }
    for(int i = 1; i < rows-1; i++){
      sendBuf[newLocal][zn][facez(i,cols-1)] = (
          recvBuf[XN][facex(i,planes-1)]+
          recvBuf[ZN][facez(i,cols-1)]+
          sendBuf[whichLocal][zn][facez(i,cols-1)]+
          sendBuf[whichLocal][zn][facez(i-1,cols-1)]+
          sendBuf[whichLocal][zn][facez(i+1,cols-1)]+
          sendBuf[whichLocal][zn][facez(i,cols-2)]+
          sendBuf[whichLocal][xn][facex(i,planes-2)]
      )/7;
      eltsComp+=1;
    }
    
    // 3. xp face
    // Interior points first
    for(int j = 1; j < planes-1; j++){
      for(int i = 1; i < rows-1; i++){
        sendBuf[newLocal][xp][facex(i,j)] = (
                sendBuf[whichLocal][xp][facex(i,j)]+
                sendBuf[whichLocal][xp][facex(i-1,j)]+
                sendBuf[whichLocal][xp][facex(i+1,j)]+
                sendBuf[whichLocal][xp][facex(i,j-1)]+
                sendBuf[whichLocal][xp][facex(i,j+1)]+
                localChunk[whichLocal][small(i-1,0,j-1)]+
                recvBuf[XP][facex(i,j)]
        )/7;
      eltsComp+=1;
      }
    }

    // Corners next
    sendBuf[newLocal][xp][facex(0,0)] = sendBuf[newLocal][zp][facez(0,0)];
    sendBuf[newLocal][xp][facex(0,planes-1)] = sendBuf[newLocal][zn][facez(0,0)];
    sendBuf[newLocal][xp][facex(rows-1,0)] = sendBuf[newLocal][zp][facez(rows-1,0)];
    sendBuf[newLocal][xp][facex(rows-1,planes-1)] = sendBuf[newLocal][zn][facez(rows-1,0)];

    // Finally, the edges: 
    // Each element here uses 2 ghosts, 4 elts from its own sendBuf, 1 elt from a different sendBuf
    
    for(int i = 1; i < planes-1; i++){
      sendBuf[newLocal][xp][facex(0,i)] = (
          recvBuf[XP][facex(0,i)]+
          recvBuf[YP][facey(0,i)]+
          sendBuf[whichLocal][xp][facex(0,i)]+
          sendBuf[whichLocal][xp][facex(0,i-1)]+
          sendBuf[whichLocal][xp][facex(0,i+1)]+
          sendBuf[whichLocal][xp][facex(1,i)]+
          sendBuf[whichLocal][yp][facey(1,i)]
      )/7;
      eltsComp+=1;
    }
    for(int i = 1; i < rows-1; i++){
      sendBuf[newLocal][xp][facex(i,0)] = sendBuf[newLocal][zp][facez(i,0)];
    }
    for(int i = 1; i < planes-1; i++){
      sendBuf[newLocal][xp][facex(rows-1,i)] = (
          recvBuf[YN][facey(0,i)]+
          recvBuf[XP][facex(rows-1,i)]+
          sendBuf[whichLocal][xp][facex(rows-1,i)]+
          sendBuf[whichLocal][xp][facex(rows-1,i-1)]+
          sendBuf[whichLocal][xp][facex(rows-1,i+1)]+
          sendBuf[whichLocal][xp][facex(rows-2,i)]+
          sendBuf[whichLocal][yn][facey(1,i)]
      )/7;
      eltsComp+=1;
    }
    for(int i = 1; i < rows-1; i++){
      sendBuf[newLocal][xp][facex(i,planes-1)] = sendBuf[newLocal][zn][facez(i,0)]; 
    }
   
    // 4. xn face
    // Interior points first
    for(int j = 1; j < planes-1; j++){
      for(int i = 1; i < rows-1; i++){
        sendBuf[newLocal][xn][facex(i,j)] = (
                sendBuf[whichLocal][xn][facex(i,j)]+
                sendBuf[whichLocal][xn][facex(i-1,j)]+
                sendBuf[whichLocal][xn][facex(i+1,j)]+
                sendBuf[whichLocal][xn][facex(i,j-1)]+
                sendBuf[whichLocal][xn][facex(i,j+1)]+
                localChunk[whichLocal][small(i-1,cols-3,j-1)]+
                recvBuf[XN][facex(i,j)]
        )/7;
      eltsComp+=1;
      }
    }

    // Corners next
    sendBuf[newLocal][xn][facex(0,0)] = sendBuf[newLocal][zp][facez(0,cols-1)];
    sendBuf[newLocal][xn][facex(0,planes-1)] = sendBuf[newLocal][zn][facez(0,cols-1)];
    sendBuf[newLocal][xn][facex(rows-1,0)] = sendBuf[newLocal][zp][facez(rows-1,cols-1)];
    sendBuf[newLocal][xn][facex(rows-1,planes-1)] = sendBuf[newLocal][zn][facez(rows-1,cols-1)];
    
    // Finally, the edges: 
    // Each element here uses 2 ghosts, 4 elts from its own sendBuf, 1 elt from a different sendBuf
    
    for(int i = 1; i < planes-1; i++){
      sendBuf[newLocal][xn][facex(0,i)] = (
          recvBuf[XN][facex(0,i)]+
          recvBuf[YP][facey(cols-1,i)]+
          sendBuf[whichLocal][xn][facex(0,i)]+
          sendBuf[whichLocal][xn][facex(0,i-1)]+
          sendBuf[whichLocal][xn][facex(0,i+1)]+
          sendBuf[whichLocal][xn][facex(1,i)]+
          sendBuf[whichLocal][yp][facey(cols-2,i)]
      )/7;
      eltsComp+=1;
    }
    for(int i = 1; i < rows-1; i++){
      sendBuf[newLocal][xn][facex(i,0)] = sendBuf[newLocal][zp][facez(i,cols-1)]; 
    }
    for(int i = 1; i < planes-1; i++){
      sendBuf[newLocal][xn][facex(rows-1,i)] = (
          recvBuf[YN][facey(cols-1,i)]+
          recvBuf[XN][facex(rows-1,i)]+
          sendBuf[whichLocal][xn][facex(rows-1,i)]+
          sendBuf[whichLocal][xn][facex(rows-1,i-1)]+
          sendBuf[whichLocal][xn][facex(rows-1,i+1)]+
          sendBuf[whichLocal][xn][facex(rows-2,i)]+
          sendBuf[whichLocal][yn][facey(cols-2,i)]
      )/7;
      eltsComp+=1;
    }
    for(int i = 1; i < rows-1; i++){
      sendBuf[newLocal][xn][facex(i,planes-1)] = sendBuf[newLocal][zn][facez(i,cols-1)];
    }
    
 
    // 5. yp face
    // Interior points first
    for(int j = 1; j < planes-1; j++){
      for(int i = 1; i < cols-1; i++){
        sendBuf[newLocal][yp][facey(i,j)] = (
                sendBuf[whichLocal][yp][facey(i,j)]+
                sendBuf[whichLocal][yp][facey(i-1,j)]+
                sendBuf[whichLocal][yp][facey(i+1,j)]+
                sendBuf[whichLocal][yp][facey(i,j-1)]+
                sendBuf[whichLocal][yp][facey(i,j+1)]+
                localChunk[whichLocal][small(0,i-1,j-1)]+
                recvBuf[YP][facey(i,j)]
        )/7;
      eltsComp+=1;
      }
    }

    // Corners next
    // Each element uses 3 ghosts, 3 from its own sendBuf (including itself)  and 1 from a different sendBuf
    sendBuf[newLocal][yp][facey(0,0)] = sendBuf[newLocal][zp][facez(0,0)];
    sendBuf[newLocal][yp][facey(0,planes-1)] = sendBuf[newLocal][zn][facez(0,0)];
    sendBuf[newLocal][yp][facey(cols-1,0)] = sendBuf[newLocal][xn][facex(0,0)];
    sendBuf[newLocal][yp][facey(cols-1,planes-1)] = sendBuf[newLocal][zn][facez(0,cols-1)]; 
    
    // Finally, the edges: 
    // Each element here uses 2 ghosts, 4 elts from its own sendBuf, 1 elt from a different sendBuf
    
    for(int i = 1; i < planes-1; i++){
      sendBuf[newLocal][yp][facey(0,i)] = sendBuf[newLocal][xp][facex(0,i)]; 
    }
    for(int i = 1; i < cols-1; i++){
      sendBuf[newLocal][yp][facey(i,0)] = sendBuf[newLocal][zp][facez(0,i)];
    }
    for(int i = 1; i < planes-1; i++){
      sendBuf[newLocal][yp][facey(cols-1,i)] = sendBuf[newLocal][xn][facex(0,i)];
    }
    for(int i = 1; i < cols-1; i++){
      sendBuf[newLocal][yp][facey(i,planes-1)] = sendBuf[newLocal][zn][facez(0,i)];
    }

    // 6. yn face
    // Interior points first
    for(int j = 1; j < planes-1; j++){
      for(int i = 1; i < cols-1; i++){
        sendBuf[newLocal][yn][facey(i,j)] = (
                sendBuf[whichLocal][yn][facey(i,j)]+
                sendBuf[whichLocal][yn][facey(i-1,j)]+
                sendBuf[whichLocal][yn][facey(i+1,j)]+
                sendBuf[whichLocal][yn][facey(i,j-1)]+
                sendBuf[whichLocal][yn][facey(i,j+1)]+
                localChunk[whichLocal][small(rows-3,i-1,j-1)]+
                recvBuf[YN][facey(i,j)]
        )/7;
      eltsComp+=1;
      }
    }

    // Corners next
    // Each element uses 3 ghosts, 3 from its own sendBuf (including itself)  and 1 from a different sendBuf
    sendBuf[newLocal][yn][facey(0,0)] = sendBuf[newLocal][zp][facez(rows-1,0)];
    sendBuf[newLocal][yn][facey(0,planes-1)] = sendBuf[newLocal][zn][facez(rows-1,0)];
    sendBuf[newLocal][yn][facey(cols-1,0)] = sendBuf[newLocal][zp][facez(rows-1,cols-1)];
    sendBuf[newLocal][yn][facey(cols-1,planes-1)] = sendBuf[newLocal][zn][facez(rows-1,cols-1)]; 
    
    // Finally, the edges: 
    // Each element here uses 2 ghosts, 4 elts from its own sendBuf, 1 elt from a different sendBuf
    
    for(int i = 1; i < planes-1; i++){
      sendBuf[newLocal][yn][facey(0,i)] = sendBuf[newLocal][xp][facex(rows-1,i)]; 
    }
    for(int i = 1; i < cols-1; i++){
      sendBuf[newLocal][yn][facey(i,0)] = sendBuf[newLocal][zp][facez(rows-1,i)];
    }
    for(int i = 1; i < planes-1; i++){
      sendBuf[newLocal][yn][facey(cols-1,i)] = sendBuf[newLocal][xn][facex(rows-1,i)];
    }
    for(int i = 1; i < cols-1; i++){
      sendBuf[newLocal][yn][facey(i,planes-1)] = sendBuf[newLocal][zn][facez(rows-1,i)];
    }


    
    
    // exclude the time for setup and the first iteration
    iterations++;
    if(iterations == 1){
      contribute(0,0,CkReduction::concat, CkCallback(CkIndex_Main::doneSetup(), mainProxy));
    }
    
    // toggle between localChunks
    whichLocal = newLocal;
    if(iterations == num_iterations){
#ifdef STENCIL2D_VERBOSE
      CkPrintf("(%d,%d): contributing to exit\n", row, col);
#endif
      contribute(sizeof(double), &eltsComp, CkReduction::sum_double, CkCallback(CkIndex_Main::done(NULL), mainProxy));
    }
    else{
#ifdef USE_CKDIRECT
      // 2. signal readiness to recv next round of data
      for(int i = 0; i < NBRS; i++){
        CkDirect_ready(&rhandles[i]);
      }
#endif
      // contribute to barrier
#ifdef STENCIL2D_VERBOSE
      CkPrintf("(%d,%d): contributing to allReady\n", row, col);
#endif
      contribute(0, 0, CkReduction::concat, CkCallback(CkIndex_StencilPoint::allReadyCallback(NULL), thisProxy));
    }
    
    if(iterations > num_iterations){
      CkPrintf("******\n(%d,%d,%d):\n******\n", row, col,plane);
      CkAbort("death is inevitable; bugs needn't be.\n");
    }
  }

  void allReadyCallback(CkReductionMsg *msg){
    delete msg;
#ifdef STENCIL2D_VERBOSE
    CkPrintf("(%d,%d,%d): all ready, send data\n", row, col,plane);
#endif
    sendData();
  }

  inline void sendMsg(int col, int row, int plane, int which, float *buf){
    StencilMsg *msg = new (payload[which/2]) StencilMsg;
    memcpy(msg->arr,buf,payload[which/2]*sizeof(float));
    msg->which = which;
    msg->size = payload[which/2];
    thisProxy[lin(col,row,plane)].recvBufferMsg(msg);
  }

  void sendData(){
    // 1. copy data into buffers from local chunk
    // top and bottom boundaries
#ifdef STENCIL2D_VERBOSE
    //CkPrintf("(%d,%d,%d): sendData() called\n", x,y,z);
#endif
#ifdef USE_CKDIRECT
    // 2. put buffers
    for(int i = 0; i < NBRS; i++){
      CkDirect_put(&shandles[whichLocal][i]);
    }
#else
#ifdef USE_MESSAGES
#ifdef ARR_CHECK
    sendMsg((col+1)%charesx,row,plane,XP,sendBuf[whichLocal][xn].getVec());
    sendMsg((col-1+charesx)%charesx,row,plane,XN,sendBuf[whichLocal][xp].getVec());
    sendMsg(col,(row+1)%charesy,plane,YP,sendBuf[whichLocal][yn].getVec());
    sendMsg(col,(row-1+charesy)%charesy,plane,YN,sendBuf[whichLocal][yp].getVec());
    sendMsg(col,row,(plane+1)%charesz,ZP,sendBuf[whichLocal][zn].getVec());
    sendMsg(col,row,(plane-1+charesz)%charesz,ZN,sendBuf[whichLocal][zp].getVec());
#else
    sendMsg((col+1)%charesx,row,plane,XP,sendBuf[whichLocal][xn]);
    sendMsg((col-1+charesx)%charesx,row,plane,XN,sendBuf[whichLocal][xp]);
    sendMsg(col,(row+1)%charesy,plane,YP,sendBuf[whichLocal][yn]);
    sendMsg(col,(row-1+charesy)%charesy,plane,YN,sendBuf[whichLocal][yp]);
    sendMsg(col,row,(plane+1)%charesz,ZP,sendBuf[whichLocal][zn]);
    sendMsg(col,row,(plane-1+charesz)%charesz,ZN,sendBuf[whichLocal][zp]);
#endif
#else
#ifdef ARR_CHECK
    // 2. send messages
    thisProxy[lin((col+1)%charesx,row,plane)].recvBuffer(sendBuf[whichLocal][xn].getVec(),payload[0],XP);
    thisProxy[lin((col-1+charesx)%charesx,row,plane)].recvBuffer(sendBuf[whichLocal][xp].getVec(),payload[0],XN);
    thisProxy[lin(col,(row+1)%charesy,plane)].recvBuffer(sendBuf[whichLocal][yn].getVec(),payload[1],YP);
    thisProxy[lin(col,(row-1+charesy)%charesy,plane)].recvBuffer(sendBuf[whichLocal][yp].getVec(),payload[1],YN);
    thisProxy[lin(col,row,(plane+1)%charesz)].recvBuffer(sendBuf[whichLocal][zn].getVec(),payload[2],ZP);
    thisProxy[lin(col,row,(plane-1+charesz)%charesz)].recvBuffer(sendBuf[whichLocal][zp].getVec(),payload[2],ZN);
#else
    thisProxy[lin((col+1)%charesx,row,plane)].recvBuffer(sendBuf[whichLocal][xn],payload[0],XP);
    thisProxy[lin((col-1+charesx)%charesx,row,plane)].recvBuffer(sendBuf[whichLocal][xp],payload[0],XN);
    thisProxy[lin(col,(row+1)%charesy,plane)].recvBuffer(sendBuf[whichLocal][yn],payload[1],YP);
    thisProxy[lin(col,(row-1+charesy)%charesy,plane)].recvBuffer(sendBuf[whichLocal][yp],payload[1],YN);
    thisProxy[lin(col,row,(plane+1)%charesz)].recvBuffer(sendBuf[whichLocal][zn],payload[2],ZP);
    thisProxy[lin(col,row,(plane-1+charesz)%charesz)].recvBuffer(sendBuf[whichLocal][zp],payload[2],ZN);
#endif
#endif // end USE_MESSAGES
#endif
  }

};

#include "stencil3d.def.h"
