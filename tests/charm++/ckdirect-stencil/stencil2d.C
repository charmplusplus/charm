#include "ckdirect.h" 
#include "stencil2d.decl.h"
/* DIMxDIM elements per chare */
#define DIM 10
#define OOB 9999999999.0
#define NBRS 6

CProxy_Main mainProxy;

int dim;
//int blockDim;
int num_iterations;

class Main : public CBase_Main {
  
  double start, end;
  public:

  Main(CkArgMsg *msg){
    mainProxy = thisProxy;

    dim = 100;
    num_iterations = 100;

    if(msg->argc == 4){
      dim = atoi(msg->argv[1]);
      blockDim = atoi(msg->argv[2]);
      num_iterations = atoi(msg->argv[3]);
    }
    delete msg;
    CProxy_StencilPoint array = 
        CProxy_StencilPoint::ckNew(dim, dim, dim);
    int charesPerDim = dim/blockDim;
    //CkPrintf("main: %dx%dx%d elt array, num_iterations: %d\n", charesPerDim, charesPerDim, charesPerDim, num_iterations);
    CkPrintf("main: %dx%dx%d elt array, num_iterations: %d\n", charesPerDim, charesPerDim, charesPerDim, num_iterations);
    start = CmiWallTimer();
    array.setupChannels();
  }

  void done(){
    end = CmiWallTimer();
    CkPrintf("Computation took %f sec\n", end-start);
    CkExit();
  }
};

class StencilPoint : public CBase_StencilPoint{
  int row, col, plane;
  int whichLocal;

  int iterations;
  int charesPerDim;
  
  /* 0: left, 1: right, 2: top, 3: bottom*/
  infiDirectUserHandle shandles[2][NBRS];
  infiDirectUserHandle rhandles[NBRS];
  /* receive buffers */
  float *recvBuf[NBRS]; /* square of side (blockDim) */
  /* send buffers */
  float *sendBuf[2][NBRS]; /* -do- */
  float *localChunk[2]; /* cube of side (blockDim-2) */

  int remainingBufs;
  int remainingChannels;
  int payload;
  
  public:
  StencilPoint(CkMigrateMessage *){}
  StencilPoint(){
    remainingBufs = NBRS;
    remainingChannels = NBRS;
    whichLocal= 0;

    iterations = 0;
    
    col = thisIndex.y;
    row = thisIndex.x;
    plane = thisIndex.z;

    charesPerDim = dim/blockDim;
    payload = blockDim*blockDim*sizeof(float);
    
    // allocate memory
    for(int i = 0; i < NBRS; i++){
      recvBuf[0][i] = new float [blockDim*blockDim];
      recvBuf[1][i] = new float [blockDim*blockDim];
      sendBuf[i] = new float [blockDim*blockDim];
    }
    
    localChunk[0] = new float[(blockDim-2)*(blockDim-2)*(blockDim-2)];
    localChunk[1] = new float[(blockDim-2)*(blockDim-2)*(blockDim-2)];
    // initialize
    if(z == 0){// face 0 is held at 0
      for(int i = 0; i < blockDim*blockDim; i++)
        // first index says which version of the double buffer is to be used
        // second index, the face in question
        sendBuf[0][0][i] = 1.0;
    }
    else{
      for(int i = 0; i < blockDim*blockDim; i++)
        // first index says which version of the double buffer is to be used
        // second index, the face in question
        sendBuf[0][0][i] = 0.0;
    }

    // rest of the domain
    // first the faces other than face 0
    for(int i = 1; i < NBRS; i++){
      for(int j = 0; j < blockDim*blockDim; j++){
        sendBuf[0][i][j] = 0.0;
      }
    }
    
    // now the rest of the cube
    for(int j = 0; j < (blockDim-2)*(blockDim-2)*(blockDim-2); j++){
      localChunk[0][j] = 0.0; 
    }
  }
  
  ~StencilPoint(){
    for(int i = 0; i < NBRS; i++){
      delete [] recvBuf[0][i];
      delete [] recvBuf[1][i];
      delete [] sendBuf[i];
    }

    delete [] localChunk[0];
    delete [] localChunk[1];
  }
  
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

  
    thisProxy((col), (row+1)%charesPerDim, (plane)).notify(node,2,thisIndex);
    thisProxy((col), (row-1+charesPerDim)%charesPerDim, (plane)).notify(node,3,thisIndex);
    thisProxy((col+1)%charesPerDim, (row), (plane)).notify(node,0,thisIndex);
    thisProxy((col-1+charesPerDim)%charesPerDim, (row), (plane)).notify(node,1,thisIndex);
    thisProxy((col), (row), (plane+1)%charesPerDim).notify(node,4,thisIndex);
    thisProxy((col), (row), (plane-1+charesPerDim)%charesPerDim).notify(node,5,thisIndex);
    /*
    thisProxy(x,y,(z+1)%charesPerDim).notify(node,4,thisIndex);
    thisProxy(x,y,(z-1+charesPerDim)%charesPerDim).notify(node,5,thisIndex);
    thisProxy((x+1)%charesPerDim,y,z).notify(node,0,thisIndex);//
    thisProxy((x-1+charesPerDim)%charesPerDim,y,z).notify(node,1,thisIndex);//
    thisProxy(x,(y+1)%charesPerDim,z).notify(node,2,thisIndex);
    thisProxy(x,(y-1+charesPerDim)%charesPerDim,z).notify(node,3,thisIndex);
    */
    /*
    thisProxy((col+1)%num_cols, row).notify(node, 0, thisIndex); 
    thisProxy((col-1+num_cols)%num_cols, row).notify(node, 1, thisIndex); 
    thisProxy(col, (row-1+num_rows)%num_rows).notify(node, 3, thisIndex);
    thisProxy(col, (row+1)%num_rows).notify(node, 2, thisIndex);
    */
  }

  void notify(int node, int which, CkIndex3D whoSent){
    // make handle
#ifdef STENCIL2D_VERBOSE
    CkPrintf("(%d,%d,%d): (%d,%d,%d) is %d to me\n", x, y, z, whoSent.y, whoSent.x, whoSent.z, which);
#endif
    rhandles[which] = CkDirect_createHandle(node, recvBuf[which], payload, StencilPoint::callbackWrapper, (void *)this, OOB);
    // i am ready to receive
    //CkPrintf("(%d,%d): calling ready on rhandles[%d]\n", row, col, which); 
    //CkDirect_ready(&rhandles[which]); 
    // send handle to sender
    thisProxy(whoSent.x, whoSent.y, whoSent.z).recvHandle(rhandles[which], which);
  }

  void recvHandle(infiDirectUserHandle handle, int which){
    // this causes an inversion of the buffer indices:
    // left send buffer is 1, right 0, top 3, bottom 2
    shandles[0][which] = handle;
    shandles[1][which] = handle;
    
    CkDirect_assocLocalBuffer(&shandles[0][which], sendBuf[0][which], payload);
    CkDirect_assocLocalBuffer(&shandles[1][which], sendBuf[1][which], payload);
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
      thisProxy(thisIndex.x, thisIndex.y, thisIndex.z).compute();
    }
  }
  
  static void callbackWrapper(void *arg){
    StencilPoint *obj = (StencilPoint *)arg;
    obj->recvBuffer();
  }
  
// to access localChunks interior (i.e. interior of interior) 
#define small(r,c,p) (p)*(blockDim-2)*(blockDim-2)+(r)*(blockDim-2)+(c)
// to access sendBufs 
#define face(r,c) (r)*blockDim+(c)
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
    for(int i = 1; i < blockDim-3; i++){
      for(int j = 1; j < blockDim-3; j++){
        for(int k = 1; k < blockDim-3; k++){
          localChunk[newLocal][small(i,j,k)] = (
                              localChunk[whichLocal][small(i,j,k)]+
                              localChunk[whichLocal][small(i+1,j,k)]+
                              localChunk[whichLocal][small(i-1,j,k)]+
                              localChunk[whichLocal][small(i,j+1,k)]+
                              localChunk[whichLocal][small(i,j-1,k)]+
                              localChunk[whichLocal][small(i,j,k+1)]+
                              localChunk[whichLocal][small(i,j,k-1)]+
                            )/7;
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
                            sendBuf[whichLocal][zp][face(1,1)]+
                            sendBuf[whichLocal][xp][face(1,1)]+
                            sendBuf[whichLocal][yp][face(1,1)]
                            )/7;

    //1.
    localChunk[newLocal][small(0,blockDim-3,0)] = (
                            localChunk[whichLocal][small(0,blockDim-3,0)]+
                            localChunk[whichLocal][small(0,blockDim-4,0)]+
                            localChunk[whichLocal][small(1,blockDim-3,0)]+
                            localChunk[whichLocal][small(0,blockDim-3,1)]+
                            sendBuf[whichLocal][zp][face(1,blockDim-2)]+
                            sendBuf[whichLocal][yp][face(blockDim-2,1)]+
                            sendBuf[whichLocal][xn][face(1,1)]
                            )/7;

    //2.
    localChunk[newLocal][small(blockDim-3,0,0)] = (
                            localChunk[whichLocal][small(blockDim-3,0,0)]+
                            localChunk[whichLocal][small(blockDim-4,0,0)]+
                            localChunk[whichLocal][small(blockDim-3,1,0)]+
                            localChunk[whichLocal][small(blockDim-3,0,1)]+
                            sendBuf[whichLocal][zp][face(blockDim-2,1)]+
                            sendBuf[whichLocal][yn][face(1,1)]+
                            sendBuf[whichLocal][xp][face(blockDim-2,1)]
                            )/7;

    //3.
    localChunk[newLocal][small(blockDim-3,blockDim-3,0)] = (
                            localChunk[whichLocal][small(blockDim-3,blockDim-3,0)]+
                            localChunk[whichLocal][small(blockDim-3,blockDim-4,0)]+
                            localChunk[whichLocal][small(blockDim-4,blockDim-3,0)]+
                            localChunk[whichLocal][small(blockDim-3,blockDim-3,1)]+
                            sendBuf[whichLocal][zp][face(blockDim-2,blockDim-2)]+
                            sendBuf[whichLocal][yn][face(blockDim-2,1)]+
                            sendBuf[whichLocal][xn][face(blockDim-2,1)]
                            )/7;
    
    //4.
    localChunk[newLocal][small(0,0,blockDim-3)] = (
                            localChunk[whichLocal][small(0,0,blockDim-3)]+
                            localChunk[whichLocal][small(0,0,blockDim-4)]+
                            localChunk[whichLocal][small(0,1,blockDim-3)]+
                            localChunk[whichLocal][small(1,0,blockDim-3)]+
                            sendBuf[whichLocal][zn][face(1,1)]+
                            sendBuf[whichLocal][yp][face(1,blockDim-2)]+
                            sendBuf[whichLocal][xp][face(1,blockDim-2)]
                            )/7;

    //5.
    localChunk[newLocal][small(0,blockDim-3,blockDim-3)] = (
                            localChunk[whichLocal][small(0,blockDim-3,blockDim-3)]+
                            localChunk[whichLocal][small(0,blockDim-4,blockDim-3)]+
                            localChunk[whichLocal][small(1,blockDim-3,blockDim-3)]+
                            localChunk[whichLocal][small(0,blockDim-3,blockDim-4)]+
                            sendBuf[whichLocal][zn][face(1,blockDim-2)]+
                            sendBuf[whichLocal][yp][face(blockDim-2,blockDim-2)]+
                            sendBuf[whichLocal][xn][face(1,blockDim-2)]
                            )/7;
    
    //6.
    localChunk[newLocal][small(blockDim-3,0,blockDim-3)] = (
                            localChunk[whichLocal][small(blockDim-3,0,blockDim-3)]+
                            localChunk[whichLocal][small(blockDim-4,0,blockDim-3)]+
                            localChunk[whichLocal][small(blockDim-3,1,blockDim-3)]+
                            localChunk[whichLocal][small(blockDim-3,0,blockDim-4)]+
                            sendBuf[whichLocal][zn][face(blockDim-2,1)]+
                            sendBuf[whichLocal][yn][face(1,blockDim-2)]+
                            sendBuf[whichLocal][xp][face(blockDim-2,blockDim-2)]
                            )/7;

    //7.
    localChunk[newLocal][small(blockDim-3,blockDim-3,blockDim-3)] = (
                            localChunk[whichLocal][small(blockDim-3,blockDim-3,blockDim-3)]+
                            localChunk[whichLocal][small(blockDim-4,blockDim-3,blockDim-3)]+
                            localChunk[whichLocal][small(blockDim-3,blockDim-4,blockDim-3)]+
                            localChunk[whichLocal][small(blockDim-3,blockDim-3,blockDim-4)]+
                            sendBuf[whichLocal][zn][face(blockDim-2,blockDim-2)]+
                            sendBuf[whichLocal][yn][face(blockDim-2,blockDim-2)]+
                            sendBuf[whichLocal][xn][face(blockDim-2,blockDim-2)]
                            )/7;


    // 12 edges
    //zp,yp
    for(int i = 1; i < blockDim-3; i++){
      localChunk[newLocal][small(0,i,0)] = (
                localChunk[whichLocal][small(0,i,0)]+
                localChunk[whichLocal][small(0,i-1,0)]+
                localChunk[whichLocal][small(0,i+1,0)]+
                localChunk[whichLocal][small(1,i,0)]+
                localChunk[whichLocal][small(0,i,1)]+
                sendBuf[whichLocal][yp][face(i,1)]+
                sendBuf[whichLocal][zp][face(1,i)]
          )/7;
    }

    //xn,yp
    for(int i = 1; i < blockDim-3; i++){
      localChunk[newLocal][small(0,blockDim-3,i)] = (
                localChunk[whichLocal][small(0,blockDim-3,i)]+
                localChunk[whichLocal][small(0,blockDim-3,i-1)]+
                localChunk[whichLocal][small(0,blockDim-3,i+1)]+
                localChunk[whichLocal][small(1,blockDim-3,i)]+
                localChunk[whichLocal][small(0,blockDim-4,i)]+
                sendBuf[whichLocal][xn][face(1,i)]+
                sendBuf[whichLocal][yp][face(blockDim-2,i)]
          )/7;
    }

    //zn,yp
    for(int i = 1; i < blockDim-3; i++){
      localChunk[newLocal][small(0,i,blockDim-3)] = (
                localChunk[whichLocal][small(0,i,blockDim-3)]+
                localChunk[whichLocal][small(0,i-1,blockDim-3)]+
                localChunk[whichLocal][small(0,i+1,blockDim-3)]+
                localChunk[whichLocal][small(0,i,blockDim-4)]+
                localChunk[whichLocal][small(1,i,blockDim-3)]+
                sendBuf[whichLocal][zn][face(1,i)]+
                sendBuf[whichLocal][yp][face(i,blockDim-2)]
          )/7;
    }

    //yp,xp
    for(int i = 1; i < blockDim-3; i++){
      localChunk[newLocal][small(0,0,i)] = (
                localChunk[whichLocal][small(0,0,i)]+
                localChunk[whichLocal][small(0,0,i-1)]+
                localChunk[whichLocal][small(0,0,i+1)]+
                localChunk[whichLocal][small(0,1,i)]+
                localChunk[whichLocal][small(1,0,i)]+
                sendBuf[whichLocal][yp][face(1,i)]+
                sendBuf[whichLocal][xp][face(1,i)]
          )/7;
    }

    //yn,zp
    for(int i = 1; i < blockDim-3; i++){
      localChunk[newLocal][small(blockDim-3,i,0)] = (
                localChunk[whichLocal][small(blockDim-3,i,0)]+
                localChunk[whichLocal][small(blockDim-3,i-1,0)]+
                localChunk[whichLocal][small(blockDim-3,i+1,0)]+
                localChunk[whichLocal][small(blockDim-4,i,0)]+
                localChunk[whichLocal][small(blockDim-3,i,1)]+
                sendBuf[whichLocal][yn][face(i,1)]+
                sendBuf[whichLocal][zp][face(blockDim-2,i)]
          )/7;
    }

    //xn,yn
    for(int i = 1; i < blockDim-3; i++){
      localChunk[newLocal][small(blockDim-3,blockDim-3,i)] = (
                localChunk[whichLocal][small(blockDim-3,blockDim-3,i)]+
                localChunk[whichLocal][small(blockDim-3,blockDim-3,i-1)]+
                localChunk[whichLocal][small(blockDim-3,blockDim-3,i+1)]+
                localChunk[whichLocal][small(blockDim-4,blockDim-3,i)]+
                localChunk[whichLocal][small(blockDim-3,blockDim-4,i)]+
                sendBuf[whichLocal][xn][face(blockDim-2,i)]+
                sendBuf[whichLocal][yn][face(blockDim-2,i)]
          )/7;
    }

    //yn,zn
    for(int i = 1; i < blockDim-3; i++){
      localChunk[newLocal][small(blockDim-3,i,blockDim-3)] = (
                localChunk[whichLocal][small(blockDim-3,i,blockDim-3)]+
                localChunk[whichLocal][small(blockDim-3,i-1,blockDim-3)]+
                localChunk[whichLocal][small(blockDim-3,i+1,blockDim-3)]+
                localChunk[whichLocal][small(blockDim-4,i,blockDim-3)]+
                localChunk[whichLocal][small(blockDim-3,i,blockDim-4)]+
                sendBuf[whichLocal][yn][face(i,blockDim-2)]+
                sendBuf[whichLocal][zn][face(blockDim-2,i)]
          )/7;
    }

    //xp,yn
    for(int i = 1; i < blockDim-3; i++){
      localChunk[newLocal][small(blockDim-3,0,i)] = (
                localChunk[whichLocal][small(blockDim-3,0,i)]+
                localChunk[whichLocal][small(blockDim-3,0,i-1)]+
                localChunk[whichLocal][small(blockDim-3,0,i+1)]+
                localChunk[whichLocal][small(blockDim-3,1,i)]+
                localChunk[whichLocal][small(blockDim-4,0,i)]+
                sendBuf[whichLocal][xp][face(blockDim-2,i)]+
                sendBuf[whichLocal][yn][face(1,i)]
          )/7;
    }

    //xp,zp
    for(int i = 1; i < blockDim-3; i++){
      localChunk[newLocal][small(i,0,0)] = (
                localChunk[whichLocal][small(i,0,0)]+
                localChunk[whichLocal][small(i-1,0,0)]+
                localChunk[whichLocal][small(i+1,0,0)]+
                localChunk[whichLocal][small(i,1,0)]+
                localChunk[whichLocal][small(i,0,1)]+
                sendBuf[whichLocal][xp][face(i,1)]+
                sendBuf[whichLocal][zp][face(i,1)]
          )/7;
    }

    //xn,zp
    for(int i = 1; i < blockDim-3; i++){
      localChunk[newLocal][small(i,blockDim-3,0)] = (
                localChunk[whichLocal][small(i,blockDim-3,0)]+
                localChunk[whichLocal][small(i-1,blockDim-3,0)]+
                localChunk[whichLocal][small(i+1,blockDim-3,0)]+
                localChunk[whichLocal][small(i,blockDim-4,0)]+
                localChunk[whichLocal][small(i,blockDim-3,1)]+
                sendBuf[whichLocal][xn][face(i,1)]+
                sendBuf[whichLocal][zp][face(i,blockDim-2)]
          )/7;
    }

    //xp,zn
    for(int i = 1; i < blockDim-3; i++){
      localChunk[newLocal][small(i,0,blockDim-3)] = (
                localChunk[whichLocal][small(i,0,blockDim-3)]+
                localChunk[whichLocal][small(i-1,0,blockDim-3)]+
                localChunk[whichLocal][small(i+1,0,blockDim-3)]+
                localChunk[whichLocal][small(i,1,blockDim-3)]+
                localChunk[whichLocal][small(i,0,blockDim-4)]+
                sendBuf[whichLocal][xp][face(i,blockDim-2)]+
                sendBuf[whichLocal][zn][face(i,1)]
          )/7;
    }

    //xn,zn
    for(int i = 1; i < blockDim-3; i++){
      localChunk[newLocal][small(i,blockDim-3,blockDim-3)] = (
                localChunk[whichLocal][small(i,blockDim-3,blockDim-3)]+
                localChunk[whichLocal][small(i-1,blockDim-3,blockDim-3)]+
                localChunk[whichLocal][small(i+1,blockDim-3,blockDim-3)]+
                localChunk[whichLocal][small(i,blockDim-4,blockDim-3)]+
                localChunk[whichLocal][small(i,blockDim-3,blockDim-4)]+
                sendBuf[whichLocal][xn][face(i,blockDim-2)]+
                sendBuf[whichLocal][zn][face(i,blockDim-2)]
          )/7;

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
    for(int i = 1; i < blockDim-1; i++){
      for(int j = 1; j < blockDim-1; j++){
        sendBuf[newLocal][zp][face(i,j)] = (
                sendBuf[whichLocal][zp][face(i,j)]+
                sendBuf[whichLocal][zp][face(i-1,j)]+
                sendBuf[whichLocal][zp][face(i+1,j)]+
                sendBuf[whichLocal][zp][face(i,j-1)]+
                sendBuf[whichLocal][zp][face(i,j+1)]+
                localChunk[whichLocal][small(i-1,j-1,0)]+
                recvBuf[ZP][face(i,j)];
        )/7;
      }
    }

    // Corners next
    // Each element uses 3 ghosts, 3 from its own sendBuf (including itself)  and 1 from a different sendBuf
    sendBuf[newLocal][zp][face(0,0)] = (
          recvBuf[YP][face(0,0)]+
          recvBuf[XP][face(0,0)]+
          recvBuf[ZP][face(0,0)]+
          sendBuf[whichLocal][zp][face(0,0)]+
          sendBuf[whichLocal][zp][face(0,1)]+
          sendBuf[whichLocal][zp][face(1,0)]+
          sendBuf[whichLocal][xp][face(0,1)]
    )/7;

    sendBuf[newLocal][zp][face(0,blockDim-1)] = (
          recvBuf[YP][face(blockDim-1,0)]+
          recvBuf[XN][face(0,0)]+
          recvBuf[ZP][face(0,blockDim-1)]+
          sendBuf[whichLocal][zp][face(0,blockDim-1)]+
          sendBuf[whichLocal][zp][face(1,blockDim-1)]+
          sendBuf[whichLocal][zp][face(0,blockDim-2)]+
          sendBuf[whichLocal][xn][face(0,1)]
    )/7;

    sendBuf[newLocal][zp][face(blockDim-1,0)] = (
          recvBuf[ZP][face(blockDim-1,0)]+
          recvBuf[XP][face(blockDim-1,0)]+
          recvBuf[YN][face(0,0)]+
          sendBuf[whichLocal][zp][face(blockDim-1,0)]+
          sendBuf[whichLocal][zp][face(blockDim-2,0)]+
          sendBuf[whichLocal][zp][face(blockDim-1,1)]+
          sendBuf[whichLocal][xp][face(blockDim-1,1)]
    )/7;

    sendBuf[newLocal][zp][face(blockDim-1,blockDim-1)] = (
          recvBuf[XN][face(blockDim-1,0)]+
          recvBuf[YN][face(blockDim-1,0)]+
          recvBuf[ZP][face(blockDim-1,blockDim-1)]+
          sendBuf[whichLocal][zp][face(blockDim-1,blockDim-1)]+
          sendBuf[whichLocal][zp][face(blockDim-2,blockDim-1)]+
          sendBuf[whichLocal][zp][face(blockDim-1,blockDim-2)]+
          sendBuf[whichLocal][xn][face(blockDim-1,1)]
    )/7;

    // Finally, the edges: 
    // Each element here uses 2 ghosts, 4 elts from its own sendBuf, 1 elt from a different sendBuf
    
    for(int i = 1; i < blockDim-1; i++){
      sendBuf[newLocal][zp][face(0,i)] = (
          recvBuf[ZP][face(0,i)]+
          recvBuf[YP][face(i,0)]+
          sendBuf[whichLocal][zp][face(0,i)]+
          sendBuf[whichLocal][zp][face(0,i-1)]+
          sendBuf[whichLocal][zp][face(0,i+1)]+
          sendBuf[whichLocal][zp][face(1,i)]+
          sendBuf[whichLocal][yp][face(i,1)]
      )/7;
    }
    for(int i = 1; i < blockDim-1; i++){
      sendBuf[newLocal][zp][face(i,0)] = (
          recvBuf[XP][face(i,0)]+
          recvBuf[ZP][face(i,0)]+
          sendBuf[whichLocal][zp][face(i,0)]+
          sendBuf[whichLocal][zp][face(i-1,0)]+
          sendBuf[whichLocal][zp][face(i+1,0)]+
          sendBuf[whichLocal][zp][face(i,1)]+
          sendBuf[whichLocal][xp][face(i,1)]
      )/7;
    }
    for(int i = 1; i < blockDim-1; i++){
      sendBuf[newLocal][zp][face(blockDim-1,i)] = (
          recvBuf[YN][face(i,0)]+
          recvBuf[ZP][face(blockDim-1,i)]+
          sendBuf[whichLocal][zp][face(blockDim-1,i)]+
          sendBuf[whichLocal][zp][face(blockDim-1,i-1)]+
          sendBuf[whichLocal][zp][face(blockDim-1,i+1)]+
          sendBuf[whichLocal][zp][face(blockDim-2,i)]+
          sendBuf[whichLocal][yn][face(i,1)]
      )/7;
    }
    for(int i = 1; i < blockDim-1; i++){
      sendBuf[newLocal][zp][face(i,blockDim-1)] = (
          recvBuf[XN][face(i,0)]+
          recvBuf[ZP][face(i,blockDim-1)]+
          sendBuf[whichLocal][zp][face(i,blockDim-1)]+
          sendBuf[whichLocal][zp][face(i-1,blockDim-1)]+
          sendBuf[whichLocal][zp][face(i+1,blockDim-1)]+
          sendBuf[whichLocal][zp][face(i,blockDim-2)]+
          sendBuf[whichLocal][xn][face(i,1)]
      )/7;
    }
    
    // 2. zn face
    // Interior points first
    for(int i = 1; i < blockDim-1; i++){
      for(int j = 1; j < blockDim-1; j++){
        sendBuf[newLocal][zn][face(i,j)] = (
                sendBuf[whichLocal][zn][face(i,j)]+
                sendBuf[whichLocal][zn][face(i-1,j)]+
                sendBuf[whichLocal][zn][face(i+1,j)]+
                sendBuf[whichLocal][zn][face(i,j-1)]+
                sendBuf[whichLocal][zn][face(i,j+1)]+
                localChunk[whichLocal][small(i-1,j-1,blockDim-3)]+
                recvBuf[ZN][face(i,j)];
        )/7;
      }
    }

    // Corners next
    // Each element uses 3 ghosts, 3 from its own sendBuf (including itself)  and 1 from a different sendBuf
    sendBuf[newLocal][zn][face(0,0)] = (
          recvBuf[YP][face(0,blockDim-1)]+
          recvBuf[XP][face(0,blockDim-1)]+
          recvBuf[ZN][face(0,0)]+
          sendBuf[whichLocal][zn][face(0,0)]+
          sendBuf[whichLocal][zn][face(0,1)]+
          sendBuf[whichLocal][zn][face(1,0)]+
          sendBuf[whichLocal][xp][face(0,blockDim-2)]
    )/7;

    sendBuf[newLocal][zn][face(0,blockDim-1)] = (
          recvBuf[YP][face(blockDim-1,blockDim-1)]+
          recvBuf[XN][face(blockDim-1,0)]+
          recvBuf[ZN][face(0,blockDim-1)]+
          sendBuf[whichLocal][zn][face(0,blockDim-1)]+
          sendBuf[whichLocal][zn][face(1,blockDim-1)]+
          sendBuf[whichLocal][zn][face(0,blockDim-2)]+
          sendBuf[whichLocal][xn][face(0,blockDim-2)]
    )/7;

    sendBuf[newLocal][zn][face(blockDim-1,0)] = (
          recvBuf[ZN][face(blockDim-1,0)]+
          recvBuf[XP][face(blockDim-1,blockDim-1)]+
          recvBuf[YN][face(0,blockDim-1)]+
          sendBuf[whichLocal][zn][face(blockDim-1,0)]+
          sendBuf[whichLocal][zn][face(blockDim-2,0)]+
          sendBuf[whichLocal][zn][face(blockDim-1,1)]+
          sendBuf[whichLocal][xp][face(blockDim-1,blockDim-2)]
    )/7;

    sendBuf[newLocal][zn][face(blockDim-1,blockDim-1)] = (
          recvBuf[XN][face(blockDim-1,blockDim-1)]+
          recvBuf[YN][face(blockDim-1,blockDim-1)]+
          recvBuf[ZN][face(blockDim-1,blockDim-1)]+
          sendBuf[whichLocal][zn][face(blockDim-1,blockDim-1)]+
          sendBuf[whichLocal][zn][face(blockDim-2,blockDim-1)]+
          sendBuf[whichLocal][zn][face(blockDim-1,blockDim-2)]+
          sendBuf[whichLocal][xn][face(blockDim-1,blockDim-2)]
    )/7;

    // Finally, the edges: 
    // Each element here uses 2 ghosts, 4 elts from its own sendBuf, 1 elt from a different sendBuf
    
    for(int i = 1; i < blockDim-1; i++){
      sendBuf[newLocal][zn][face(0,i)] = (
          recvBuf[ZN][face(0,i)]+
          recvBuf[YP][face(i,blockDim-1)]+
          sendBuf[whichLocal][zn][face(0,i)]+
          sendBuf[whichLocal][zn][face(0,i-1)]+
          sendBuf[whichLocal][zn][face(0,i+1)]+
          sendBuf[whichLocal][zn][face(1,i)]+
          sendBuf[whichLocal][yp][face(i,blockDim-2)]
      )/7;
    }
    for(int i = 1; i < blockDim-1; i++){
      sendBuf[newLocal][zn][face(i,0)] = (
          recvBuf[XP][face(i,blockDim-1)]+
          recvBuf[ZN][face(i,0)]+
          sendBuf[whichLocal][zn][face(i,0)]+
          sendBuf[whichLocal][zn][face(i-1,0)]+
          sendBuf[whichLocal][zn][face(i+1,0)]+
          sendBuf[whichLocal][zn][face(i,1)]+
          sendBuf[whichLocal][xp][face(i,blockDim-2)]
      )/7;
    }
    for(int i = 1; i < blockDim-1; i++){
      sendBuf[newLocal][zn][face(blockDim-1,i)] = (
          recvBuf[YN][face(i,blockDim-1)]+
          recvBuf[ZN][face(blockDim-1,i)]+
          sendBuf[whichLocal][zn][face(blockDim-1,i)]+
          sendBuf[whichLocal][zn][face(blockDim-1,i-1)]+
          sendBuf[whichLocal][zn][face(blockDim-1,i+1)]+
          sendBuf[whichLocal][zn][face(blockDim-2,i)]+
          sendBuf[whichLocal][yn][face(i,blockDim-2)]
      )/7;
    }
    for(int i = 1; i < blockDim-1; i++){
      sendBuf[newLocal][zn][face(i,blockDim-1)] = (
          recvBuf[XN][face(i,blockDim-1)]+
          recvBuf[ZN][face(i,blockDim-1)]+
          sendBuf[whichLocal][zn][face(i,blockDim-1)]+
          sendBuf[whichLocal][zn][face(i-1,blockDim-1)]+
          sendBuf[whichLocal][zn][face(i+1,blockDim-1)]+
          sendBuf[whichLocal][zn][face(i,blockDim-2)]+
          sendBuf[whichLocal][xn][face(i,blockDim-2)]
      )/7;
    }
    
    // 1. xp face
    // Interior points first
    for(int i = 1; i < blockDim-1; i++){
      for(int j = 1; j < blockDim-1; j++){
        sendBuf[newLocal][zp][face(i,j)] = (
                sendBuf[whichLocal][zp][face(i,j)]+
                sendBuf[whichLocal][zp][face(i-1,j)]+
                sendBuf[whichLocal][zp][face(i+1,j)]+
                sendBuf[whichLocal][zp][face(i,j-1)]+
                sendBuf[whichLocal][zp][face(i,j+1)]+
                //FIXME - localChunk[whichLocal][small(i-1,j-1)]+
                recvBuf[ZP][face(i,j)];
        )/7;
      }
    }

    // Corners next
    // Each element uses 3 ghosts, 3 from its own sendBuf (including itself)  and 1 from a different sendBuf
    sendBuf[newLocal][zp][face(0,0)] = (
          recvBuf[YP][face(0,0)]+
          recvBuf[XP][face(0,0)]+
          recvBuf[ZP][face(0,0)]+
          sendBuf[whichLocal][zp][face(0,0)]+
          sendBuf[whichLocal][zp][face(0,1)]+
          sendBuf[whichLocal][zp][face(1,0)]+
          sendBuf[whichLocal][xp][face(0,1)]
    )/7;

    sendBuf[newLocal][zp][face(0,blockDim-1)] = (
          recvBuf[YP][face(blockDim-1,0)]+
          recvBuf[XN][face(0,0)]+
          recvBuf[ZP][face(0,blockDim-1)]+
          sendBuf[whichLocal][zp][face(0,blockDim-1)]+
          sendBuf[whichLocal][zp][face(1,blockDim-1)]+
          sendBuf[whichLocal][zp][face(0,blockDim-2)]+
          sendBuf[whichLocal][xn][face(0,1)]
    )/7;

    sendBuf[newLocal][zp][face(blockDim-1,0)] = (
          recvBuf[ZP][face(blockDim-1,0)]+
          recvBuf[XP][face(blockDim-1,0)]+
          recvBuf[YN][face(0,0)]+
          sendBuf[whichLocal][zp][face(blockDim-1,0)]+
          sendBuf[whichLocal][zp][face(blockDim-2,0)]+
          sendBuf[whichLocal][zp][face(blockDim-1,1)]+
          sendBuf[whichLocal][xp][face(blockDim-1,1)]
    )/7;

    sendBuf[newLocal][zp][face(blockDim-1,blockDim-1)] = (
          recvBuf[XN][face(blockDim-1,0)]+
          recvBuf[YN][face(blockDim-1,0)]+
          recvBuf[ZP][face(blockDim-1,blockDim-1)]+
          sendBuf[whichLocal][zp][face(blockDim-1,blockDim-1)]+
          sendBuf[whichLocal][zp][face(blockDim-2,blockDim-1)]+
          sendBuf[whichLocal][zp][face(blockDim-1,blockDim-2)]+
          sendBuf[whichLocal][xn][face(blockDim-1,1)]
    )/7;

    // Finally, the edges: 
    // Each element here uses 2 ghosts, 4 elts from its own sendBuf, 1 elt from a different sendBuf
    
    for(int i = 1; i < blockDim-1; i++){
      sendBuf[newLocal][zp][face(0,i)] = (
          recvBuf[ZP][face(0,i)]+
          recvBuf[YP][face(i,0)]+
          sendBuf[whichLocal][zp][face(0,i)]+
          sendBuf[whichLocal][zp][face(0,i-1)]+
          sendBuf[whichLocal][zp][face(0,i+1)]+
          sendBuf[whichLocal][zp][face(1,i)]+
          sendBuf[whichLocal][yp][face(i,1)]
      )/7;
    }
    for(int i = 1; i < blockDim-1; i++){
      sendBuf[newLocal][zp][face(i,0)] = (
          recvBuf[XP][face(i,0)]+
          recvBuf[ZP][face(i,0)]+
          sendBuf[whichLocal][zp][face(i,0)]+
          sendBuf[whichLocal][zp][face(i-1,0)]+
          sendBuf[whichLocal][zp][face(i+1,0)]+
          sendBuf[whichLocal][zp][face(i,1)]+
          sendBuf[whichLocal][xp][face(i,1)]
      )/7;
    }
    for(int i = 1; i < blockDim-1; i++){
      sendBuf[newLocal][zp][face(blockDim-1,i)] = (
          recvBuf[YN][face(i,0)]+
          recvBuf[ZP][face(blockDim-1,i)]+
          sendBuf[whichLocal][zp][face(blockDim-1,i)]+
          sendBuf[whichLocal][zp][face(blockDim-1,i-1)]+
          sendBuf[whichLocal][zp][face(blockDim-1,i+1)]+
          sendBuf[whichLocal][zp][face(blockDim-2,i)]+
          sendBuf[whichLocal][yn][face(i,1)]
      )/7;
    }
    for(int i = 1; i < blockDim-1; i++){
      sendBuf[newLocal][zp][face(i,blockDim-1)] = (
          recvBuf[XN][face(i,0)]+
          recvBuf[ZP][face(i,blockDim-1)]+
          sendBuf[whichLocal][zp][face(i,blockDim-1)]+
          sendBuf[whichLocal][zp][face(i-1,blockDim-1)]+
          sendBuf[whichLocal][zp][face(i+1,blockDim-1)]+
          sendBuf[whichLocal][zp][face(i,blockDim-2)]+
          sendBuf[whichLocal][xn][face(i,1)]
      )/7;
    }
    
    // 1. zp face
    // Interior points first
    for(int i = 1; i < blockDim-1; i++){
      for(int j = 1; j < blockDim-1; j++){
        sendBuf[newLocal][zp][face(i,j)] = (
                sendBuf[whichLocal][zp][face(i,j)]+
                sendBuf[whichLocal][zp][face(i-1,j)]+
                sendBuf[whichLocal][zp][face(i+1,j)]+
                sendBuf[whichLocal][zp][face(i,j-1)]+
                sendBuf[whichLocal][zp][face(i,j+1)]+
                localChunk[whichLocal][small(i-1,j-1)]+
                recvBuf[ZP][face(i,j)];
        )/7;
      }
    }

    // Corners next
    // Each element uses 3 ghosts, 3 from its own sendBuf (including itself)  and 1 from a different sendBuf
    sendBuf[newLocal][zp][face(0,0)] = (
          recvBuf[YP][face(0,0)]+
          recvBuf[XP][face(0,0)]+
          recvBuf[ZP][face(0,0)]+
          sendBuf[whichLocal][zp][face(0,0)]+
          sendBuf[whichLocal][zp][face(0,1)]+
          sendBuf[whichLocal][zp][face(1,0)]+
          sendBuf[whichLocal][xp][face(0,1)]
    )/7;

    sendBuf[newLocal][zp][face(0,blockDim-1)] = (
          recvBuf[YP][face(blockDim-1,0)]+
          recvBuf[XN][face(0,0)]+
          recvBuf[ZP][face(0,blockDim-1)]+
          sendBuf[whichLocal][zp][face(0,blockDim-1)]+
          sendBuf[whichLocal][zp][face(1,blockDim-1)]+
          sendBuf[whichLocal][zp][face(0,blockDim-2)]+
          sendBuf[whichLocal][xn][face(0,1)]
    )/7;

    sendBuf[newLocal][zp][face(blockDim-1,0)] = (
          recvBuf[ZP][face(blockDim-1,0)]+
          recvBuf[XP][face(blockDim-1,0)]+
          recvBuf[YN][face(0,0)]+
          sendBuf[whichLocal][zp][face(blockDim-1,0)]+
          sendBuf[whichLocal][zp][face(blockDim-2,0)]+
          sendBuf[whichLocal][zp][face(blockDim-1,1)]+
          sendBuf[whichLocal][xp][face(blockDim-1,1)]
    )/7;

    sendBuf[newLocal][zp][face(blockDim-1,blockDim-1)] = (
          recvBuf[XN][face(blockDim-1,0)]+
          recvBuf[YN][face(blockDim-1,0)]+
          recvBuf[ZP][face(blockDim-1,blockDim-1)]+
          sendBuf[whichLocal][zp][face(blockDim-1,blockDim-1)]+
          sendBuf[whichLocal][zp][face(blockDim-2,blockDim-1)]+
          sendBuf[whichLocal][zp][face(blockDim-1,blockDim-2)]+
          sendBuf[whichLocal][xn][face(blockDim-1,1)]
    )/7;

    // Finally, the edges: 
    // Each element here uses 2 ghosts, 4 elts from its own sendBuf, 1 elt from a different sendBuf
    
    for(int i = 1; i < blockDim-1; i++){
      sendBuf[newLocal][zp][face(0,i)] = (
          recvBuf[ZP][face(0,i)]+
          recvBuf[YP][face(i,0)]+
          sendBuf[whichLocal][zp][face(0,i)]+
          sendBuf[whichLocal][zp][face(0,i-1)]+
          sendBuf[whichLocal][zp][face(0,i+1)]+
          sendBuf[whichLocal][zp][face(1,i)]+
          sendBuf[whichLocal][yp][face(i,1)]
      )/7;
    }
    for(int i = 1; i < blockDim-1; i++){
      sendBuf[newLocal][zp][face(i,0)] = (
          recvBuf[XP][face(i,0)]+
          recvBuf[ZP][face(i,0)]+
          sendBuf[whichLocal][zp][face(i,0)]+
          sendBuf[whichLocal][zp][face(i-1,0)]+
          sendBuf[whichLocal][zp][face(i+1,0)]+
          sendBuf[whichLocal][zp][face(i,1)]+
          sendBuf[whichLocal][xp][face(i,1)]
      )/7;
    }
    for(int i = 1; i < blockDim-1; i++){
      sendBuf[newLocal][zp][face(blockDim-1,i)] = (
          recvBuf[YN][face(i,0)]+
          recvBuf[ZP][face(blockDim-1,i)]+
          sendBuf[whichLocal][zp][face(blockDim-1,i)]+
          sendBuf[whichLocal][zp][face(blockDim-1,i-1)]+
          sendBuf[whichLocal][zp][face(blockDim-1,i+1)]+
          sendBuf[whichLocal][zp][face(blockDim-2,i)]+
          sendBuf[whichLocal][yn][face(i,1)]
      )/7;
    }
    for(int i = 1; i < blockDim-1; i++){
      sendBuf[newLocal][zp][face(i,blockDim-1)] = (
          recvBuf[XN][face(i,0)]+
          recvBuf[ZP][face(i,blockDim-1)]+
          sendBuf[whichLocal][zp][face(i,blockDim-1)]+
          sendBuf[whichLocal][zp][face(i-1,blockDim-1)]+
          sendBuf[whichLocal][zp][face(i+1,blockDim-1)]+
          sendBuf[whichLocal][zp][face(i,blockDim-2)]+
          sendBuf[whichLocal][xn][face(i,1)]
      )/7;
    }
    


  

    for(int i = 1; i < DIM-1; i++){
      for(int j = 1; j < DIM-1; j++){
        localChunk[newLocal][i][j] = (localChunk[whichLocal][i+1][j] + localChunk[whichLocal][i-1][j] + localChunk[whichLocal][i][j+1] + localChunk[whichLocal][i][j-1] + localChunk[whichLocal][i][j])/5;
      }
    }
    
    // corners
    if(row != 0){
      localChunk[newLocal][0][0] = (localChunk[whichLocal][0][0]+
                                  localChunk[whichLocal][1][0]+
                                  localChunk[whichLocal][0][1]+
                                  recvBuf[2][0]+recvBuf[0][0])/5;
      localChunk[newLocal][0][DIM-1] = (localChunk[whichLocal][0][DIM-1]+
                                      localChunk[whichLocal][0][DIM-2]+
                                      localChunk[whichLocal][1][DIM-1]+
                                      recvBuf[1][0]+recvBuf[2][DIM-1])/5;
    }
    localChunk[newLocal][DIM-1][0] = (localChunk[whichLocal][DIM-1][0]+
                                      localChunk[whichLocal][DIM-2][0]+
                                      localChunk[whichLocal][DIM-1][1]+
                                      recvBuf[3][0]+recvBuf[0][DIM-1])/5;
    localChunk[newLocal][DIM-1][DIM-1] = (localChunk[whichLocal][DIM-1][DIM-1]+
                                          localChunk[whichLocal][DIM-2][DIM-1]+
                                          localChunk[whichLocal][DIM-1][DIM-2]+
                                          recvBuf[3][DIM-1]+recvBuf[1][DIM-1])/5;
    
    if(row != 0){
      //boundaries: top and bottom
      for(int i = 1; i < DIM-1; i++){
        localChunk[newLocal][0][i] = (localChunk[whichLocal][0][i]+
                                    localChunk[whichLocal][0][i-1]+
                                    localChunk[whichLocal][0][i+1]+
                                    localChunk[whichLocal][1][i]+
                                    recvBuf[2][i])/5; 
        localChunk[newLocal][DIM-1][i] = (localChunk[whichLocal][DIM-1][i]+
                                        localChunk[whichLocal][DIM-1][i-1]+
                                        localChunk[whichLocal][DIM-1][i+1]+
                                        localChunk[whichLocal][DIM-2][i]+
                                        recvBuf[3][i])/5;
      }
    }
    else{
      for(int i = 0; i < DIM; i++){
        localChunk[newLocal][0][i] = 1.0;
      }
    }

    //boundaries: left and right
    for(int i = 1; i < DIM-1; i++){
      localChunk[newLocal][i][0] = (localChunk[whichLocal][i][0]+
                                    recvBuf[0][i]+
                                    localChunk[whichLocal][i][1]+
                                    localChunk[whichLocal][i-1][0]+
                                    localChunk[whichLocal][i+1][0])/5;
      localChunk[newLocal][i][DIM-1] = (localChunk[whichLocal][i][DIM-1] + 
                                        recvBuf[1][i]+
                                        localChunk[whichLocal][i][DIM-2]+
                                        localChunk[whichLocal][i-1][DIM-1]+
                                        localChunk[whichLocal][i+1][DIM-1])/5;
    }
    
    // toggle between localChunks
    whichLocal = newLocal;
    iterations++;
    if(iterations == num_iterations){
      CkCallback cb(CkIndex_Main::done(), mainProxy);
#ifdef STENCIL2D_VERBOSE
      CkPrintf("(%d,%d): contributing to exit\n", row, col);
#endif
      contribute(0,0,CkReduction::concat, cb);
    }
    else{
      // 2. signal readiness to recv next round of data
      for(int i = 0; i < NBRS; i++){
        CkDirect_ready(&rhandles[i]);
      }
      // contribute to barrier
      CkCallback cb(CkIndex_StencilPoint::allReadyCallback(NULL), thisProxy);
#ifdef STENCIL2D_VERBOSE
      CkPrintf("(%d,%d): contributing to allReady\n", row, col);
#endif
      contribute(0, 0, CkReduction::concat, cb);
    }
    if(iterations > num_iterations){
      CkPrintf("******\n(%d,%d):\n******\n", row, col);
      CkAbort("death is inevitable; bugs needn't be.\n");
    }
  }

  void allReadyCallback(CkReductionMsg *msg){
    delete msg;
#ifdef STENCIL2D_VERBOSE
    CkPrintf("(%d,%d): all ready, send data\n", row, col);
#endif
    sendData();
  }

  void sendData(){
    // 1. copy data into buffers from local chunk
    // top and bottom boundaries
#ifdef STENCIL2D_VERBOSE
    CkPrintf("(%d,%d,%d): sendData() called\n", x,y,z);
#endif
    /*
    for(int i = 0; i < DIM; i++){
      sendBuf[3][i] = localChunk[whichLocal][0][i];
      sendBuf[2][i] = localChunk[whichLocal][DIM-1][i];
    }   
    // left and right boundaries
    for(int i = 0; i < DIM; i++){
      sendBuf[1][i] = localChunk[whichLocal][i][0];
      sendBuf[0][i] = localChunk[whichLocal][i][DIM-1];
    }
    */
    // 2. put buffers
    for(int i = 0; i < NBRS; i++){
      CkDirect_put(&shandles[whichLocal][i]);
    }
  }

};

#include "stencil2d.def.h"
