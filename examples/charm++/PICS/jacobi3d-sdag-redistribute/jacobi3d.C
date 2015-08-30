#define LEFT			1
#define RIGHT			2
#define TOP			3
#define BOTTOM			4
#define FRONT			5
#define BACK			6
#define DIVIDEBY7       	0.14285714285714285714
#define DELTA       	        0.01

#define TUNE 1

//#define  FASTMAP    1
//#define  HILBERTMAP  1
#include "jacobi3d.decl.h"
#if TUNE
#include "picsautoperfAPI.h"
#include "picsautoperfAPIC.h"
#include "picsautotunerAPI.h"
#endif


/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int arrayDimX;
/*readonly*/ int arrayDimY;
/*readonly*/ int arrayDimZ;
/*readonly*/ int blockDimX;
/*readonly*/ int blockDimY;
/*readonly*/ int blockDimZ;

// specify the number of worker chares in each dimension
/*readonly*/ int num_chare_x;
/*readonly*/ int num_chare_y;
/*readonly*/ int num_chare_z;

#define wrapX(a)	(((a)+num_chare_x)%num_chare_x)
#define wrapY(a)	(((a)+num_chare_y)%num_chare_y)
#define wrapZ(a)	(((a)+num_chare_z)%num_chare_z)

#define index(a,b,c)	((a)+(b)*(blockDimX+2)+(c)*(blockDimX+2)*(blockDimY+2))
#define index3(a,b,c,aa,bb)	((a)+(b)*(aa)+(c)*(aa)*(bb))

double startTime;
double endTime;

/** \class Main
 *
 */

class redistributeMsg : public CMessage_redistributeMsg{

public:
    int x, y, z;
    int splitter;
    double *data;

    redistributeMsg(int x1, int y1, int z1, int sp)
    {
        x = x1;
        y = y1;
        z = z1;
        splitter = sp;
    }
};


class Main : public CBase_Main {
public:
    int max_iters;
    double maxdifference;
    int blockDimX;
    int blockDimY;
    int blockDimZ;
    int num_chare_x;
    int num_chare_y;
    int num_chare_z;
    CProxy_Jacobi array;
    CProxy_Jacobi array2;

    int initstatus;
    int old_x, old_y, old_z;
    int iterations;
    double factor;
    Main_SDAG_CODE;
    Main(CkArgMsg* m) {
        if ( (m->argc != 3) && (m->argc != 7) && (m->argc != 4) && (m->argc != 8)) {
            CkPrintf("%s [array_size] [block_size]\n", m->argv[0]);
            CkPrintf("OR %s [array_size_X] [array_size_Y] [array_size_Z] [block_size_X] [block_size_Y] [block_size_Z]\n", m->argv[0]);
            CkAbort("Abort");
        }

        // set iteration counter to zero
        iterations = 0;
        max_iters = 10;
        // store the main proxy
        mainProxy = thisProxy;

        if(m->argc <5 ) {
            arrayDimX = arrayDimY = arrayDimZ = atoi(m->argv[1]);
            blockDimX = blockDimY = blockDimZ = atoi(m->argv[2]);
            if(m->argc == 4)
                max_iters =  atoi(m->argv[3]);
        }
        else if (m->argc <9) {
            arrayDimX = atoi(m->argv[1]);
            arrayDimY = atoi(m->argv[2]);
            arrayDimZ = atoi(m->argv[3]);
            blockDimX = atoi(m->argv[4]);
            blockDimY = atoi(m->argv[5]);
            blockDimZ = atoi(m->argv[6]);
            if(m->argc == 8)
                max_iters =  atoi(m->argv[7]);
        }

        if (arrayDimX < blockDimX || arrayDimX % blockDimX != 0)
            CkAbort("array_size_X % block_size_X != 0!");
        if (arrayDimY < blockDimY || arrayDimY % blockDimY != 0)
            CkAbort("array_size_Y % block_size_Y != 0!");
        if (arrayDimZ < blockDimZ || arrayDimZ % blockDimZ != 0)
            CkAbort("array_size_Z % block_size_Z != 0!");

        num_chare_x = arrayDimX / blockDimX;
        num_chare_y = arrayDimY / blockDimY;
        num_chare_z = arrayDimZ / blockDimZ;

        // print info
        CkPrintf("\nSTENCIL COMPUTATION WITH NO BARRIERS\n");
        CkPrintf("%f Running Jacobi on %d processors with (%d, %d, %d) chares\n", CkNumPes(), num_chare_x, num_chare_y, num_chare_z);
        CkPrintf("Array Dimensions: %d %d %d\n", arrayDimX, arrayDimY, arrayDimZ);
        CkPrintf("Block Dimensions: %d %d %d\n", blockDimX, blockDimY, blockDimZ);

        // Create new array of worker chares
        array = CProxy_Jacobi::ckNew(num_chare_x, num_chare_y, num_chare_z, blockDimX, blockDimY, blockDimZ, num_chare_x,  num_chare_y, num_chare_z);

        initstatus = 1;
        startTime = CkWallTimer();
        thisProxy.registerTune();
    }


    void newRun()
    {
        double factor;
        int old_x = num_chare_x;
        int old_y = num_chare_y;
        int old_z = num_chare_z;
        int old_blockDimX = blockDimX;
        int old_blockDimY = blockDimY;
        int old_blockDimZ = blockDimZ;
#if TUNE
        int valid;
        blockDimX = PICS_getTunedParameter("blockDimX", &valid);
        blockDimY = PICS_getTunedParameter("blockDimY", &valid);
        blockDimZ = PICS_getTunedParameter("blockDimZ", &valid);
#endif
        factor = (double)old_blockDimX/blockDimX;
        num_chare_x = arrayDimX/blockDimX;
        num_chare_y = arrayDimY/blockDimY;
        num_chare_z = arrayDimY/blockDimZ;
#if TUNE
        if(initstatus == 1){
            array = CProxy_Jacobi::ckNew(num_chare_x, num_chare_y, num_chare_z, blockDimX, blockDimY, blockDimZ, num_chare_x,  num_chare_y, num_chare_z);
            thisProxy.run();
            initstatus = 0;
        }else
        {
            CkPrintf(" factor %f\n", factor);
            array2 = CProxy_Jacobi::ckNew(num_chare_x, num_chare_y, num_chare_z, blockDimX, blockDimY, blockDimZ, num_chare_x,  num_chare_y, num_chare_z);
            array.redistribute(array2, factor);
        }
#else

        if(initstatus == 1){
#if FASTMAP
            CProxy_FastArrayMap mymap = CProxy_FastArrayMap::ckNew();
#elif HILBERTMAP
            CProxy_HilbertArrayMap mymap = CProxy_HilbertArrayMap::ckNew();
#elif BLOCKMAP
            CProxy_BlockMap mymap = CProxy_BlockMap::ckNew();
#else
            CProxy_DefaultArrayMap mymap = CProxy_DefaultArrayMap::ckNew();
#endif
            CkArrayOptions opts (num_chare_x, num_chare_y, num_chare_z);
            opts.setMap(mymap);
            array = CProxy_Jacobi::ckNew(num_chare_x, num_chare_y, num_chare_z, blockDimX, blockDimY, blockDimZ, opts);
        }
        thisProxy.run();
        initstatus = 0;
#endif

  }

  void registerTune()
  {
#if TUNE
      PICS_registerTunableParameterFields("blockDimX", TP_INT, blockDimX, 16, arrayDimX, 2, PICS_EFF_GRAINSIZE, 1, OP_MUL, TS_PERF_GUIDE, 1);
      PICS_registerTunableParameterFields("blockDimY", TP_INT, blockDimY, 16, arrayDimY, 2, PICS_EFF_GRAINSIZE, 1, OP_MUL, TS_PERF_GUIDE, 1);
      PICS_registerTunableParameterFields("blockDimZ", TP_INT, blockDimZ, 16, arrayDimZ, 2, PICS_EFF_GRAINSIZE, 1, OP_MUL, TS_PERF_GUIDE, 1);
#endif
      thisProxy.newRun();
  }

  void redistributeDone() {
      int i,j,k;
      for(i=0; i<old_x; i++)
          for(j=0;j<old_y;j++)
              for(k=0;k<old_z;k++)
                  array(i,j,k).ckDestroy();
      array = array2;
      thisProxy.run();
  }


  void done(bool success) {
      if(iterations < max_iters )
      {
          thisProxy.newRun();
          //thisProxy.run();
      }else
      {

          CkPrintf("Completed %d Iterations , Difference %lf fails threshhold\n", iterations,maxdifference);
          endTime = CkWallTimer();
          CkPrintf("Time elapsed per iteration: %f total duration %f \n", (endTime - startTime)/(max_iters), endTime - startTime);
          CkExit();
      }
  }

};

/** \class Jacobi
 *
 */

class Jacobi: public CBase_Jacobi {
  Jacobi_SDAG_CODE

public:
  int iterations;
  int neighbors;
  int remoteCount;
  double error;
  double *temperature;
  double *new_temperature;
  bool converged;


  int blockDimX;
  int blockDimY;
  int blockDimZ;
  int num_chare_x;
  int num_chare_y;
  int num_chare_z;
  int recvChildren;

  // Constructor, initialize values
  Jacobi(int cx, int cy, int cz, int bx, int by, int bz) {

      recvChildren = 0;
      num_chare_x = cx;
      num_chare_y = cy;
      num_chare_z = cz;
      blockDimX = bx;
      blockDimY = by;
      blockDimZ = bz;
      // allocate a three dimensional array
      temperature = new double[(blockDimX+2) * (blockDimY+2) * (blockDimZ+2)];
      new_temperature = new double[(blockDimX+2) * (blockDimY+2) * (blockDimZ+2)];

      for(int k=0; k<blockDimZ+2; ++k)
          for(int j=0; j<blockDimY+2; ++j)
              for(int i=0; i<blockDimX+2; ++i)
                  new_temperature[index(i, j, k)] = temperature[index(i, j, k)] = 0.0;
      iterations = 0;
      commonInit();
  }


  void commonInit()
  {
      converged = false;
      neighbors = 6;
      if(thisIndex.x == 0)
          neighbors--;
      if( thisIndex.x== num_chare_x-1)
          neighbors--;
      if(thisIndex.y == 0)
          neighbors--;
      if( thisIndex.y== num_chare_y-1)
          neighbors--;
      if(thisIndex.z == 0)
          neighbors--;
      if( thisIndex.z== num_chare_z-1)
          neighbors--;

      constrainBC();
  }

  void redistribute(CProxy_Jacobi newarray, double splitFactor) {
      beginTuneOverhead();
      int i, j, k;
      int ii, jj, kk;
      int factor;
      int subX, subY, subZ;
      int subStartX, subStartY, subStartZ;
      int childBlockDimX, childBlockDimY, childBlockDimZ;

      if(splitFactor > 1 ) { //split
          factor = (int)splitFactor;
          childBlockDimX = blockDimX/factor;
          childBlockDimY = blockDimY/factor;
          childBlockDimZ = blockDimZ/factor;
          for(int i=0; i< factor; i++)
          {
              for(int j=0; j<factor; j++)
              {
                  for(int k=0; k<factor; k++)
                  {
                      redistributeMsg* subMsg = new( childBlockDimX*childBlockDimY*childBlockDimZ) redistributeMsg(0, 0, 0, 1);
                      subStartX = i * childBlockDimX;
                      subStartY = j * childBlockDimY;
                      subStartZ = k * childBlockDimZ;
                      for(ii=1; ii<childBlockDimX+1; ++ii) {
                          for(jj=1; jj<childBlockDimY+1; ++jj) {
                              for(kk=1; kk<childBlockDimY+1; ++kk) {
                                  subMsg->data[index3(ii-1, jj-1, kk-1, childBlockDimX, childBlockDimY)] = temperature[index(ii+subStartX, jj+subStartY, kk+subStartZ)];
                              }
                          }
                      }
                      newarray(thisIndex.x*factor+i, thisIndex.y*factor+j, thisIndex.z*factor+k).construct(subMsg);
                  }
              }
          }
      }else if (splitFactor <= 1) //merge
      {
          //calculate children index
          factor = (int)(1/splitFactor);
          redistributeMsg* subMsg = new(blockDimX*blockDimY*blockDimZ) redistributeMsg(thisIndex.x%factor, thisIndex.y%factor, thisIndex.z%factor, factor);
          for(i=1; i<blockDimX+1; ++i) {
              for(j=1; j<blockDimY+1; ++j) {
                  for(k=1; k<blockDimY+1; ++k) {
                      subMsg->data[index3(i-1, j-1, k-1, blockDimX, blockDimY)] = temperature[index(i, j, k)];
                  }
              }
          }
          newarray(thisIndex.x/factor, thisIndex.y/factor, thisIndex.z/factor).construct(subMsg);
      }
      endTuneOverhead();
  }


  void construct(redistributeMsg* msg)
  {
      beginTuneOverhead();
      int i,j,k;
      int subBlockDimX = blockDimX/msg->splitter;
      int subBlockDimY = blockDimY/msg->splitter;
      int subBlockDimZ = blockDimZ/msg->splitter;
      int subX = msg->x*subBlockDimX;
      int subY = msg->y*subBlockDimY;
      int subZ = msg->z*subBlockDimZ;

      for(i=1; i<subBlockDimX+1; ++i) {
          for(j=1; j<subBlockDimY+1; ++j) {
              for(k=1; k<subBlockDimZ+1; ++k) {
              temperature[index(i+subX, j+subY, k+subZ)] = msg->data[index3(i-1, j-1, k-1, subBlockDimX, subBlockDimY)];
              }
          }
      }
      recvChildren++;
      if(recvChildren == msg->splitter * msg->splitter *  msg->splitter )
      {
          CkCallback cb(CkReductionTarget(Main, redistributeDone), mainProxy);
          contribute(0, NULL, CkReduction::nop, cb);
          recvChildren = 0;
      }
      delete msg;
      endTuneOverhead();
  }


  void pup(PUP::er &p)
  {
    CBase_Jacobi::pup(p);
    __sdag_pup(p);
    p|iterations;
    p|neighbors;

    size_t size = (blockDimX+2) * (blockDimY+2) * (blockDimZ+2);
    if (p.isUnpacking()) {
      temperature = new double[size];
      new_temperature = new double[size];
    }
    p(temperature, size);
    p(new_temperature, size);
  }

  Jacobi(CkMigrateMessage* m) { }

  ~Jacobi() {
    delete [] temperature;
    delete [] new_temperature;
  }

  // Send ghost faces to the six neighbors
  void begin_iteration(void) {
    // Copy different faces into messages
    double *leftGhost =  new double[blockDimY*blockDimZ];
    double *rightGhost =  new double[blockDimY*blockDimZ];
    double *topGhost =  new double[blockDimX*blockDimZ];
    double *bottomGhost =  new double[blockDimX*blockDimZ];
    double *frontGhost =  new double[blockDimX*blockDimY];
    double *backGhost =  new double[blockDimX*blockDimY];
    for(int k=0; k<blockDimZ; ++k)
      for(int j=0; j<blockDimY; ++j) {
        leftGhost[k*blockDimY+j] = temperature[index(1, j+1, k+1)];
        rightGhost[k*blockDimY+j] = temperature[index(blockDimX, j+1, k+1)];
      }

    for(int k=0; k<blockDimZ; ++k)
      for(int i=0; i<blockDimX; ++i) {
        topGhost[k*blockDimX+i] = temperature[index(i+1, 1, k+1)];
        bottomGhost[k*blockDimX+i] = temperature[index(i+1, blockDimY, k+1)];
      }

    for(int j=0; j<blockDimY; ++j)
      for(int i=0; i<blockDimX; ++i) {
        frontGhost[j*blockDimX+i] = temperature[index(i+1, j+1, 1)];
        backGhost[j*blockDimX+i] = temperature[index(i+1, j+1, blockDimZ)];
      }

    int x = thisIndex.x, y = thisIndex.y, z = thisIndex.z;
    if(thisIndex.x>0)
        thisProxy(wrapX(x-1),y,z).updateGhosts(iterations, RIGHT,  blockDimY, blockDimZ, rightGhost);
    if(thisIndex.x<num_chare_x-1)
        thisProxy(wrapX(x+1),y,z).updateGhosts(iterations, LEFT,   blockDimY, blockDimZ, leftGhost);
    if(thisIndex.y>0)
        thisProxy(x,wrapY(y-1),z).updateGhosts(iterations, TOP,    blockDimX, blockDimZ, topGhost);
    if(thisIndex.y<num_chare_y-1)
        thisProxy(x,wrapY(y+1),z).updateGhosts(iterations, BOTTOM, blockDimX, blockDimZ, bottomGhost);
    if(thisIndex.z>0)
        thisProxy(x,y,wrapZ(z-1)).updateGhosts(iterations, BACK,   blockDimX, blockDimY, backGhost);
    if(thisIndex.z<num_chare_z-1)
        thisProxy(x,y,wrapZ(z+1)).updateGhosts(iterations, FRONT,  blockDimX, blockDimY, frontGhost);

    delete [] leftGhost;
    delete [] rightGhost;
    delete [] bottomGhost;
    delete [] topGhost;
    delete [] frontGhost;
    delete [] backGhost;
  }

  void updateBoundary(int dir, int height, int width, double* gh) {
    switch(dir) {
    case LEFT:
      for(int k=0; k<width; ++k)
        for(int j=0; j<height; ++j) {
          temperature[index(0, j+1, k+1)] = gh[k*height+j];
        }
      break;
    case RIGHT:
      for(int k=0; k<width; ++k)
        for(int j=0; j<height; ++j) {
          temperature[index(blockDimX+1, j+1, k+1)] = gh[k*height+j];
        }
      break;
    case BOTTOM:
      for(int k=0; k<width; ++k)
        for(int i=0; i<height; ++i) {
          temperature[index(i+1, 0, k+1)] = gh[k*height+i];
        }
      break;
    case TOP:
      for(int k=0; k<width; ++k)
        for(int i=0; i<height; ++i) {
          temperature[index(i+1, blockDimY+1, k+1)] = gh[k*height+i];
        }
      break;
    case FRONT:
      for(int j=0; j<width; ++j)
        for(int i=0; i<height; ++i) {
          temperature[index(i+1, j+1, 0)] = gh[j*height+i];
        }
      break;
    case BACK:
      for(int j=0; j<width; ++j)
        for(int i=0; i<height; ++i) {
          temperature[index(i+1, j+1, blockDimZ+1)] = gh[j*height+i];
        }
      break;
    default:
      CkAbort("ERROR\n");
    }
  }

  // Check to see if we have received all neighbor values yet
  // If all neighbor values have been received, we update our values and proceed
  double computeKernel() {
    double error = 0.0, max_error = 0.0;
    for(int k=1; k<blockDimZ+1; ++k)
      for(int j=1; j<blockDimY+1; ++j)
        for(int i=1; i<blockDimX+1; ++i) {
          // update my value based on the surrounding values
          new_temperature[index(i, j, k)] = (temperature[index(i-1, j, k)]
                                             +  temperature[index(i+1, j, k)]
                                             +  temperature[index(i, j-1, k)]
                                             +  temperature[index(i, j+1, k)]
                                             +  temperature[index(i, j, k-1)]
                                             +  temperature[index(i, j, k+1)]
                                             +  temperature[index(i, j, k)] ) * DIVIDEBY7;
          error = fabs(new_temperature[index(i,j,k)] - temperature[index(i,j,k)]);
          if (error > max_error) {
            max_error = error;
          }
        } // end for

    double *tmp;
    tmp = temperature;
    temperature = new_temperature;
    new_temperature = tmp;

    //constrainBC();

    return max_error;
  }

  void print()
  {

    for(int k=1; k<blockDimZ+2; ++k)
      for(int j=1; j<blockDimY+2; ++j)
        for(int i=1; i<blockDimX+2; ++i)
          CkPrintf(" -%d:%d:%d %f ", k,j,i, temperature[index(k, j, i)]);
    CkPrintf("--------------------------------\n");
  }
  // Enforce some boundary conditions
  void constrainBC() {
    // // Heat right, left
    if(thisIndex.x == 0 )
        for(int j=0; j<blockDimY+2; ++j)
            for(int k=0; k<blockDimZ+2; ++k)
            {
                new_temperature[index(0, j, k)] = temperature[index(0, j, k)] = 255.0;
            }
    if(thisIndex.y == 0 )
        for(int j=0; j<blockDimX+2; ++j)
            for(int k=0; k<blockDimZ+2; ++k)
            {
                new_temperature[index(j,0, k)]  = temperature[index(j,0, k)] = 255.0;
            }
    if(thisIndex.z == 0 )
        for(int j=0; j<blockDimX+2; ++j)
            for(int k=0; k<blockDimY+2; ++k)
            {
                new_temperature[index(j, k, 0)] = temperature[index(j, k, 0)] = 255.0;
            }

  }
};

#include "jacobi3d.def.h"
