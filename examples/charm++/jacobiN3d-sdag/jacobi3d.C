#include <map>
#include <queue>
#include <vector>

#define MAX_ITER		26
#define LEFT			1
#define RIGHT			2
#define TOP			3
#define BOTTOM			4
#define FRONT			5
#define BACK			6
#define DIVIDEBY7       	0.14285714285714285714
#define DELTA       	        0.01
#define LBPERIOD       	        50
#define CHECKPOINTPERIOD        500
#define BUFFER_LIMIT		6 // Min Value of Limit = 6

#include "jacobi3d.decl.h"

using namespace std;

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_GhostBuffer ghostProxy;
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

#define index(a,b,c)	((a)+(b)*(blockDimX)+(c)*(blockDimX)*(blockDimY))
#define mp(a,b,c,d)     make_pair(a,make_pair(b,make_pair(c,d)))

struct messageStruct
{
        int x,y,z;
        int dir;
        int pe;
        int iter;
	bool save;
};

struct chareRequestContainer
{
        queue <messageStruct> q;
        int iter;
};


double startTime;
double endTime;

/** \class Main
 *
 */
class Main : public CBase_Main {
public:
  CProxy_Jacobi array;
  int iterations;

  Main(CkArgMsg* m) {
    if ( (m->argc != 3) && (m->argc != 7) ) {
      CkPrintf("%s [array_size] [block_size]\n", m->argv[0]);
      CkPrintf("OR %s [array_size_X] [array_size_Y] [array_size_Z] [block_size_X] [block_size_Y] [block_size_Z]\n", m->argv[0]);
      CkAbort("Abort");
    }

    // set iteration counter to zero
    iterations = 0;

    // store the main proxy
    mainProxy = thisProxy;
	
    if(m->argc == 3) {
      arrayDimX = arrayDimY = arrayDimZ = atoi(m->argv[1]);
      blockDimX = blockDimY = blockDimZ = atoi(m->argv[2]); 
    }
    else if (m->argc == 7) {
      arrayDimX = atoi(m->argv[1]);
      arrayDimY = atoi(m->argv[2]);
      arrayDimZ = atoi(m->argv[3]);
      blockDimX = atoi(m->argv[4]); 
      blockDimY = atoi(m->argv[5]); 
      blockDimZ = atoi(m->argv[6]);
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
    CkPrintf("Running Jacobi on %d processors with (%d, %d, %d) chares\n", CkNumPes(), num_chare_x, num_chare_y, num_chare_z);
    CkPrintf("Array Dimensions: %d %d %d\n", arrayDimX, arrayDimY, arrayDimZ);
    CkPrintf("Block Dimensions: %d %d %d\n", blockDimX, blockDimY, blockDimZ);

    // Create new array of worker chares
    array = CProxy_Jacobi::ckNew(num_chare_x, num_chare_y, num_chare_z);

    ghostProxy = CProxy_GhostBuffer::ckNew();

    //Start the computation
    array.run();
    startTime = CkWallTimer();
  }

  void done(int iterations) {
    CkPrintf("Completed %d iterations\n", iterations);
    endTime = CkWallTimer();
    CkPrintf("Time elapsed per iteration: %f\n", (endTime - startTime) / iterations);
    CkExit();
  }
};

class GhostBuffer: public CBase_GhostBuffer {
	map <pair <int, pair <int,pair <int , int > > > , double* > bufferMap; 
	messageStruct tmp;
	chareRequestContainer pack;
        struct comp : public std::binary_function<chareRequestContainer*, chareRequestContainer*, bool>
        {
                bool operator() (const chareRequestContainer*  x, const chareRequestContainer*  y) const {
                        return x->iter > y->iter;
                }
        };

        priority_queue  <chareRequestContainer*, vector <chareRequestContainer*>, comp > requestQueue;

public:
	int numberOfBuffer;
	GhostBuffer() {
		numberOfBuffer = 0;
	}

        void setBuffer(CProxy_Jacobi proxy, int w,int h,int iterations,int x,int y,int z, int dir, double ghost[],int save)
        {
		if(save == true)
		{
			double *ghostValue = new double[w*h];
			for(int i=0;i<h;i++)
				for(int j=0;j<w;j++)
					ghostValue[i*w+j] = ghost[i*w+j];

			bufferMap[mp(x,y,z,dir)] = ghostValue;
		}
                if(dir == RIGHT)                                                     
			proxy(wrapX(x-1),y,z).updateGhosts();
                if(dir == LEFT)                                                      
			proxy(wrapX(x+1),y,z).updateGhosts();
                if(dir == TOP)
			proxy(x,wrapY(y-1),z).updateGhosts();
                if(dir == BOTTOM)                                                    
			proxy(x,wrapY(y+1),z).updateGhosts();
                if(dir == BACK)                                                    
			proxy(x,y,wrapZ(z-1)).updateGhosts();
                if(dir == FRONT)                                                    
			proxy(x,y,wrapZ(z+1)).updateGhosts();
		
        }                                                                            

        void requestBuffer(CProxy_Jacobi proxy, chareRequestContainer* packMessage)
        {
                requestQueue.push(packMessage);
                processQueueRequests(proxy);
        }

        void releaseBuffer(int numReleased, CProxy_Jacobi proxy)
        {
                numberOfBuffer -= numReleased; 
		//CkPrintf("num of buffer released: %d %d\n",numberOfBuffer,bufferMap.size());
                processQueueRequests(proxy);
        }

        void processQueueRequests(CProxy_Jacobi proxy)
        {
                while (numberOfBuffer < BUFFER_LIMIT && requestQueue.size()!=0)
                {
                        numberOfBuffer++;
			
                        chareRequestContainer *packTmp = requestQueue.top();
                        tmp = packTmp->q.front();packTmp->q.pop();
                        if (packTmp->q.size() == 0)
                        {
                                delete requestQueue.top();
                                requestQueue.pop();
                        }
			if(tmp.dir == RIGHT || tmp.dir == LEFT)
				proxy(wrapX(tmp.x),tmp.y,tmp.z).requestMessage(tmp.iter,tmp.pe,tmp.dir,tmp.save);
			if(tmp.dir == TOP || tmp.dir == BOTTOM)
				proxy(tmp.x,wrapY(tmp.y),tmp.z).requestMessage(tmp.iter,tmp.pe,tmp.dir,tmp.save);
			if(tmp.dir == BACK || tmp.dir == FRONT)
				proxy(tmp.x,tmp.y,wrapZ(tmp.z)).requestMessage(tmp.iter,tmp.pe,tmp.dir,tmp.save);
                }
		//CkPrintf("num of buffer set: %d\n",numberOfBuffer);
        }


        double* getBuffer(int dir,int x,int y,int z)
        {
		return bufferMap[mp(x,y,z,dir)];
        }

	void deleteBuffer(int dir,int x,int y,int z)
	{
		delete [] bufferMap[mp(x,y,z,dir)];
		bufferMap.erase(mp(x,y,z,dir));
	}
};


/** \class Jacobi
 *
 */

class Jacobi: public CBase_Jacobi {
  Jacobi_SDAG_CODE

public:
  int iterations;
  int remoteCount,remoteCount_2;

  double *temperature;
  double *new_temperature;
  bool converged;
  bool done_compute;

  // Constructor, initialize values
  Jacobi() {
    usesAtSync = true;
    converged = false;
    // allocate a three dimensional array
    temperature = new double[(blockDimX) * (blockDimY) * (blockDimZ)];
    new_temperature = new double[(blockDimX) * (blockDimY) * (blockDimZ)];

    for(int k=0; k<blockDimZ; ++k)
      for(int j=0; j<blockDimY; ++j)
        for(int i=0; i<blockDimX; ++i)
          temperature[index(i, j, k)] = 0.0;

    iterations = 0;
    constrainBC();
  }

  void pup(PUP::er &p)
  {
    CBase_Jacobi::pup(p);
    __sdag_pup(p);
    p|iterations;
    p|remoteCount;
    p|remoteCount_2;

    size_t size = (blockDimX) * (blockDimY) * (blockDimZ);
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

    int x = thisIndex.x, y = thisIndex.y, z = thisIndex.z;
    int pe = CkMyPe();

    queue <messageStruct> chareRequestsQueue;


    messageStruct right;
    right.x=x+1; right.y=y; right.z=z; right.iter=iterations; right.pe=pe; right.dir=RIGHT;
    if(x==num_chare_x-1)
	    right.save = false;
    else 
	    right.save = true;
    chareRequestsQueue.push(right);

    messageStruct left;
    left.x=x-1; left.y=y; left.z=z; left.iter=iterations; left.pe=pe; left.dir=LEFT;
    if(x==0)
	    left.save = false;
    else 
	    left.save = true;
    chareRequestsQueue.push(left);

    messageStruct top;
    top.x=x; top.y=y+1; top.z=z; top.iter=iterations; top.pe=pe; top.dir=TOP;
    if(y==num_chare_y-1)
	    top.save = false;
    else 
	    top.save = true;
    chareRequestsQueue.push(top);

    messageStruct bottom;
    bottom.x=x; bottom.y=y-1; bottom.z=z; bottom.iter=iterations; bottom.pe=pe; bottom.dir=BOTTOM;
    if(y==0)
	    bottom.save = false;
    else 
	    bottom.save = true;
    chareRequestsQueue.push(bottom);

    messageStruct back;
    back.x=x; back.y=y; back.z=z+1; back.iter=iterations; back.pe=pe; back.dir=BACK;
    if(z==num_chare_z-1)
	    back.save = false;
    else 
	    back.save = true;
    chareRequestsQueue.push(back);

    messageStruct front;
    front.x=x; front.y=y; front.z=z-1; front.iter=iterations; front.pe=pe; front.dir=FRONT;
    if(z==0)
	    front.save = false;
    else 
	    front.save = true;
    chareRequestsQueue.push(front);


    chareRequestContainer *packMessage = new chareRequestContainer();
    packMessage->q = chareRequestsQueue;
    packMessage->iter = iterations;
    ghostProxy.ckLocalBranch()->requestBuffer(thisProxy,packMessage);

  }


void send(int dir,int pe,int iter,int save)
{
	int x = thisIndex.x;
	int y = thisIndex.y;
	int z = thisIndex.z;

	if(dir == LEFT)
	{
		double *leftGhost =  new double[blockDimY*blockDimZ];
		for(int k=0; k<blockDimZ; ++k)
			for(int j=0; j<blockDimY; ++j) 
				leftGhost[k*blockDimY+j] = temperature[index(0, j, k)];
		ghostProxy[pe].setBuffer(thisProxy,blockDimY,blockDimZ,iter,x,y,z,LEFT, leftGhost,save);
		delete [] leftGhost;
	}

	if(dir == TOP)
	{
		double *topGhost =  new double[blockDimX*blockDimZ];
		for(int k=0; k<blockDimZ; ++k)
			for(int i=0; i<blockDimX; ++i)
				topGhost[k*blockDimX+i] = temperature[index(i, 0, k)];
		ghostProxy[pe].setBuffer(thisProxy,blockDimX,blockDimZ,iter,x,y,z,TOP, topGhost,save);
		delete [] topGhost;
	}

	if(dir == FRONT)
	{
		double *frontGhost =  new double[blockDimX*blockDimY];
		for(int j=0; j<blockDimY; ++j)
			for(int i=0; i<blockDimX; ++i)
				frontGhost[j*blockDimX+i] = temperature[index(i, j, 0)];
		ghostProxy[pe].setBuffer(thisProxy,blockDimY,blockDimX,iter,x,y,z,FRONT, frontGhost,save);
		delete [] frontGhost;
	}

	if(dir == RIGHT)
	{
		double *rightGhost =  new double[blockDimY*blockDimZ];
		for(int k=0; k<blockDimZ; ++k)
			for(int j=0; j<blockDimY; ++j)
				rightGhost[k*blockDimY+j] = temperature[index(blockDimX-1, j, k)];
		ghostProxy[pe].setBuffer(thisProxy,blockDimY,blockDimZ,iter,x,y,z,RIGHT, rightGhost,save);
		delete [] rightGhost;
	}

	if(dir == BOTTOM)
	{
		double *bottomGhost =  new double[blockDimX*blockDimZ];
		for(int k=0; k<blockDimZ; ++k)
			for(int i=0; i<blockDimX; ++i)
				bottomGhost[k*blockDimX+i] = temperature[index(i, blockDimY-1, k)];
		ghostProxy[pe].setBuffer(thisProxy,blockDimX,blockDimZ,iter,x,y,z,BOTTOM, bottomGhost,save);
		delete [] bottomGhost;
	}

	if(dir == BACK)
	{
		double *backGhost =  new double[blockDimX*blockDimY];
		for(int j=0; j<blockDimY; ++j)
			for(int i=0; i<blockDimX; ++i)
				backGhost[j*blockDimX+i] = temperature[index(i, j, blockDimZ-1)];
		ghostProxy[pe].setBuffer(thisProxy,blockDimY,blockDimX,iter,x,y,z,BACK, backGhost,save);
		delete [] backGhost;
	}
}

  // Check to see if we have received all neighbor values yet
  // If all neighbor values have been received, we update our values and proceed
  double computeKernel() {
#pragma unroll    
    int x = thisIndex.x;
    int y = thisIndex.y;
    int z = thisIndex.z;
    double *left,*right,*top,*bottom,*front,*back;
    double leftVal,rightVal,topVal,bottomVal,frontVal,backVal;

    if(x!=0)
	    left   = ghostProxy.ckLocalBranch()->getBuffer(LEFT,x-1,y,z);
    if(y!=0)
	    bottom = ghostProxy.ckLocalBranch()->getBuffer(BOTTOM,x,y-1,z);
    if(z!=0)
	    front  = ghostProxy.ckLocalBranch()->getBuffer(FRONT,x,y,z-1);
    if(x!=num_chare_x-1)
	    right  = ghostProxy.ckLocalBranch()->getBuffer(RIGHT,x+1,y,z);
    if(y!=num_chare_y-1)
	    top    = ghostProxy.ckLocalBranch()->getBuffer(TOP,x,y+1,z);
    if(z!=num_chare_z-1)
	    back   = ghostProxy.ckLocalBranch()->getBuffer(BACK,x,y,z+1);

    for(int k=0; k<blockDimZ; ++k)
      for(int j=0; j<blockDimY; ++j)
        for(int i=0; i<blockDimX; ++i) {
          // update my value based on the surrounding values
	if(i==0)
	{
		if(x!=0)
		leftVal = left[(k)*blockDimY+(j)];
		else
		leftVal = temperature[index(i,j,k)];
	}
	else
		leftVal = temperature[index(i-1,j,k)];

	if(j==0)
	{	if(y!=0)
		bottomVal = bottom[(k)*blockDimX+(i)];
		else
		bottomVal = temperature[index(i,j,k)];
	}
	else
		bottomVal = temperature[index(i,j-1,k)];

	if(k==0)
	{
		if(z!=0)
		frontVal = front[(j)*blockDimX+(i)];
		else
		frontVal = temperature[index(i,j,k)];
	}
	else
		frontVal = temperature[index(i,j,k-1)];

	if(i==blockDimX-1)
	{
		if(x!=num_chare_x-1)
		rightVal = right[(k)*blockDimY+(j)];
		else
		rightVal = temperature[index(i,j,k)];
	}
	else
		rightVal = temperature[index(i+1,j,k)];

	if(j==blockDimY-1)
	{
		if(y!=num_chare_y-1)
		topVal = top[(k)*blockDimX+(i)];
		else
		//topVal = temperature[index(i,j+1,k)];
		topVal = 255.0; // TODO: Hack for consistency with existing results
	}
	else
		topVal = temperature[index(i,j+1,k)];

	if(k==blockDimZ-1)
	{
		if(z!=num_chare_z-1)
		backVal = back[(j)*blockDimX+(i)];
		else
		backVal = temperature[index(i,j,k)];
	}
	else
		backVal = temperature[index(i,j,k+1)];

          new_temperature[index(i, j, k)] = (   leftVal
                                             +  rightVal
					     +  bottomVal
					     +  topVal
					     +  frontVal
                                             +  backVal
                                             +  temperature[index(i, j, k)] ) * DIVIDEBY7;
        } // end for
    
    double error = 0.0, max_error = 0.0;

    for(int k=0; k<blockDimZ; ++k)
      for(int j=0; j<blockDimY; ++j)
        for(int i=0; i<blockDimX; ++i) {
          error = fabs(new_temperature[index(i,j,k)] - temperature[index(i,j,k)]);
          if (error > max_error) {
            max_error = error;
          }
        }


    if(x!=0)
	    ghostProxy.ckLocalBranch()->deleteBuffer(LEFT,x-1,y,z);
    if(y!=0)
	    ghostProxy.ckLocalBranch()->deleteBuffer(BOTTOM,x,y-1,z);
    if(z!=0)
	    ghostProxy.ckLocalBranch()->deleteBuffer(FRONT,x,y,z-1);
    if(x!=num_chare_x-1)
	    ghostProxy.ckLocalBranch()->deleteBuffer(RIGHT,x+1,y,z);
    if(y!=num_chare_y-1)
	    ghostProxy.ckLocalBranch()->deleteBuffer(TOP,x,y+1,z);
    if(z!=num_chare_z-1)
	    ghostProxy.ckLocalBranch()->deleteBuffer(BACK,x,y,z+1);

    return max_error;
  }

  // Enforce some boundary conditions
  void constrainBC() {
    // Heat left, top and front faces of each chare's block
    for(int k=0; k<blockDimZ; ++k)
      for(int i=0; i<blockDimX; ++i)
        temperature[index(i, 0, k)] = 255.0;
    for(int k=0; k<blockDimZ; ++k)
      for(int j=0; j<blockDimY; ++j)
        temperature[index(0, j, k)] = 255.0;
    for(int j=0; j<blockDimY; ++j)
      for(int i=0; i<blockDimX; ++i)
        temperature[index(i, j, 0)] = 255.0;
  }

};

#include "jacobi3d.def.h"
