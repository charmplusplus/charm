/** \file jacobi2d.C
 *  Author: Eric Bohm and Abhinav S Bhatele
 *
 *  This is jacobi3d-sdag cut down to 2d and fixed to be a correct
 *  implementation of the finite difference method by Eric Bohm.
 *
 *  Date Created: Dec 7th, * 2010
 *  
 *  Modified: Ajay Nair
 *  
 *  This is memory optimized 2d computation where chare buffer memory 
 *  is not initilized for computations. Instead a shared buffer memory
 *  is being used by chares belonging to the same group.
 */

#include "jacobi2d.decl.h"
#include <vector>
#include <map>
#include <queue>

using namespace std;
// See README for documentation

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_GhostBuffer ghostProxy;
/*readonly*/ int arrayDimX;
/*readonly*/ int arrayDimY;
/*readonly*/ int blockDimX;
/*readonly*/ int blockDimY;

// specify the number of worker chares in each dimension
/*readonly*/ int num_chare_x;
/*readonly*/ int num_chare_y;

/*readonly*/ int maxiterations;
static unsigned long next = 1;

struct messageStruct
{
	int x,y;
	int dir;
	int pe;
	int iter;
};

struct chareRequestContainer
{
	queue <messageStruct> q;
	int iter;
};

#define index(a, b)	( (b)*(blockDimX) + (a) )
#define mp(a,b) 	make_pair(a,b)

#define MAX_ITER		100
#define WARM_ITER		2
#define LEFT			1
#define RIGHT			2
#define TOP			3
#define BOTTOM			4
#define DIVIDEBY5       	0.2
#define DISPLAY_FLAG		1
#define BUFFER_LIMIT		4

const double THRESHHOLD =  0.001;

double startTime;
double endTime;

/** \class Main
 *
 */
class Main : public CBase_Main {
public:
  CProxy_Jacobi array;
  int iterations;
  double maxdifference;
  
  Main_SDAG_CODE;
  Main(CkArgMsg* m) {
    if ( (m->argc < 3) || (m->argc > 6)) {
      CkPrintf("%s [array_size] [block_size]\n", m->argv[0]);
      CkPrintf("OR %s [array_size] [block_size] maxiterations\n", m->argv[0]);
      CkPrintf("OR %s [array_size_X] [array_size_Y] [block_size_X] [block_size_Y] \n", m->argv[0]);
      CkPrintf("OR %s [array_size_X] [array_size_Y] [block_size_X] [block_size_Y] maxiterations\n", m->argv[0]);
      CkAbort("Abort");
    }

    // set iteration counter to zero
    iterations = 0;
    // store the main proxy
    mainProxy = thisProxy;
	
    if(m->argc <= 4) {
      arrayDimX = arrayDimY = atoi(m->argv[1]);
      blockDimX = blockDimY = atoi(m->argv[2]);
    }
    else if (m->argc >= 5) {
      arrayDimX = atoi(m->argv[1]);
      arrayDimY = atoi(m->argv[2]);
      blockDimX = atoi(m->argv[3]); 
      blockDimY = atoi(m->argv[4]); 
    }
    maxiterations=MAX_ITER;
    if(m->argc==4)
      maxiterations=atoi(m->argv[3]); 
    if(m->argc==6)
      maxiterations=atoi(m->argv[5]); 
      
    if (arrayDimX < blockDimX || arrayDimX % blockDimX != 0)
      CkAbort("array_size_X % block_size_X != 0!");
    if (arrayDimY < blockDimY || arrayDimY % blockDimY != 0)
      CkAbort("array_size_Y % block_size_Y != 0!");

    num_chare_x = arrayDimX / blockDimX;
    num_chare_y = arrayDimY / blockDimY;

    // print info
    CkPrintf("\nSTENCIL COMPUTATION WITH NO BARRIERS\n");
    CkPrintf("Running Jacobi on %d processors with (%d, %d) chares\n", CkNumPes(), num_chare_x, num_chare_y);
    CkPrintf("Array Dimensions: %d %d\n", arrayDimX, arrayDimY);
    CkPrintf("Block Dimensions: %d %d\n", blockDimX, blockDimY);
    CkPrintf("max iterations %d\n", maxiterations);
    CkPrintf("Threshhold %.10g\n", THRESHHOLD);
      
    // NOTE: boundary conditions must be set based on values
      
    // make proxy and populate array in one call
    ghostProxy = CProxy_GhostBuffer::ckNew();
    array = CProxy_Jacobi::ckNew(num_chare_x, num_chare_y);
    // initiate computation
    thisProxy.run();
  }

  void done(bool success) {
      if(success)
	CkPrintf("Difference %.10g Satisfied Threshhold %.10g in %d Iterations\n", maxdifference,THRESHHOLD,iterations);
      else
	CkPrintf("Completed %d Iterations , Difference %lf fails threshhold\n", iterations,maxdifference);
      endTime = CkWallTimer();
      CkPrintf("Time elapsed per iteration: %f\n", (endTime - startTime)/(maxiterations-1-WARM_ITER));
      CkExit();
  }

};

class GhostBuffer: public CBase_GhostBuffer {
	map <pair <int, pair <int ,int> > ,double* > bufferMap; // x,y,data -> to be sent
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
		numberOfBuffer= 0;
	}

	void setBuffer(CProxy_Jacobi proxy, int size,int iterations,int x, int y, int dir, double ghost[])
	{
		double *ghostValue = new double[size];
		for(int i=0;i<size;i++)
			ghostValue[i] = ghost[i];

		bufferMap[mp(dir,mp(x,y))] = ghostValue;

		if(dir == TOP)
			proxy(x,y+1).receiveGhosts();//,CkPrintf("Setting %d %d %d\n",x,y+1,dir);

		if(dir == BOTTOM)
			proxy(x,y-1).receiveGhosts();//,CkPrintf("Setting %d %d %d\n",x,y-1,dir);

		if(dir == LEFT)
			proxy(x+1,y).receiveGhosts();//,CkPrintf("Setting %d %d %d\n",x+1,y,dir);

		if(dir == RIGHT)
			proxy(x-1,y).receiveGhosts();//,CkPrintf("Setting %d %d %d\n",x-1,y,dir);
	}

	void requestBuffer(CProxy_Jacobi proxy, chareRequestContainer* packMessage)
	{
		requestQueue.push(packMessage);
		processQueueRequests(proxy);
	}

	void releaseBuffer(int numReleased, CProxy_Jacobi proxy)
	{
		numberOfBuffer -= numReleased;
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
			proxy(tmp.x,tmp.y).requestMessage(tmp.iter,tmp.pe,tmp.dir);
		}
	}

	double* getBuffer(int dir,int x,int y)
	{
		return bufferMap[mp(dir,mp(x,y))];
	}
	

	void deleteBuffer(int dir,int x,int y)
	{
		delete [] bufferMap[mp(dir,mp(x,y))];
		//bufferMap.erase(mp(dir,mp(x,y)));
	}

};

/** \class Jacobi
 *
 */

class Jacobi: public CBase_Jacobi {
  Jacobi_SDAG_CODE

  public:
  double *temperature;
  double *new_temperature;
  int imsg,itmp;
  int iterations;
  int numExpected;
  int istart,ifinish,jstart,jfinish;
  double maxdifference;
  bool leftBound, rightBound, topBound, bottomBound;
  bool done_compute;

  // Constructor, initialize values
  Jacobi() {
    usesAtSync=true;

    int i, j;
    // allocate a two dimensional array
    temperature = new double[(blockDimX) * (blockDimY)];
    memset(temperature, 0, sizeof(double)*blockDimX*blockDimY);
    new_temperature = new double[(blockDimX) * (blockDimY)];

    imsg=0;
    iterations = 0;
    numExpected=0;
    maxdifference=0.;
    // determine border conditions
    leftBound=rightBound=topBound=bottomBound=false;
    istart=jstart=0;
    ifinish=blockDimX;
    jfinish=blockDimY;

    if(thisIndex.x==0)
     {
	leftBound=true;
	istart++;
      }
    else
      numExpected++;

    if(thisIndex.x==num_chare_x-1)
      {
	rightBound=true;
	ifinish--;
      }
    else
      numExpected++;

    if(thisIndex.y==0)
      {
	topBound=true;
	jstart++;
      }
    else
      numExpected++;

    if(thisIndex.y==num_chare_y-1)
      {
	bottomBound=true;
	jfinish--;
      }
    else
      numExpected++;
    constrainBC();

    //display();
  }

  void display()
  {
    if (DISPLAY_FLAG == 0 && !(thisIndex.x == 0 && thisIndex.y == 0)) return;
    ckout<<"Iteration: "<<iterations<<" Called ("<<thisIndex.x<<","<<thisIndex.y<<") "<<" PE: "<<CkMyPe()<< " chare:\n";
    for(int i=0; i<blockDimX; ++i,ckout<<endl) {
      for(int j=0; j<blockDimY; ++j) {
	ckout<<temperature[index(i,j)]<<" ";
      } 
    }
  }
  void pup(PUP::er &p)
  {
    CBase_Jacobi::pup(p);
    __sdag_pup(p);
    p|imsg;
    p|iterations;
    p|numExpected;
    p|maxdifference;
    p|istart; p|ifinish; p|jstart; p|jfinish;
    p|leftBound; p|rightBound; p|topBound; p|bottomBound;
    
    size_t size = (blockDimX) * (blockDimY);
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
    AtSync();
    if (thisIndex.x == 0 && thisIndex.y == 0) {
      CkPrintf("Start of iteration %d\n", iterations);
    }
    iterations++;


    int x = thisIndex.x;
    int y = thisIndex.y;

    queue <messageStruct> chareRequestsQueue;
   
    if(!leftBound)
    {
      messageStruct left;
      left.x = x-1; left.y = y; left.pe = CkMyPe(); left.dir = LEFT; left.iter = iterations;
      chareRequestsQueue.push(left);
    }

    if(!rightBound)
    {
      messageStruct right;
      right.x = x+1; right.y = y; right.pe = CkMyPe(); right.dir = RIGHT; right.iter = iterations;
      chareRequestsQueue.push(right);
    }
      
    if(!topBound)
    {
      messageStruct top;
      top.x = x; top.y = y-1; top.pe = CkMyPe(); top.dir = TOP; top.iter = iterations;
      chareRequestsQueue.push(top);
    }
      
    if(!bottomBound)
    {
      messageStruct bottom;
      bottom.x = x; bottom.y = y+1; bottom.pe = CkMyPe(); bottom.dir = BOTTOM; bottom.iter = iterations;
      chareRequestsQueue.push(bottom);
    }

    chareRequestContainer *packMessage = new chareRequestContainer();
    packMessage->q = chareRequestsQueue;
    packMessage->iter = iterations;
    ghostProxy.ckLocalBranch()->requestBuffer(thisProxy,packMessage);
      
  }

  void send(int dir, int pe,int iter)
  {
	if(dir == RIGHT) 
	{
		double *leftGhost =  new double[blockDimY];
		for(int j=0; j<blockDimY; ++j) 
			leftGhost[j] = temperature[index(0, j)];
		ghostProxy[pe].setBuffer(thisProxy,blockDimY,iter,thisIndex.x,thisIndex.y,RIGHT, leftGhost);
		delete [] leftGhost;
	}

	if(dir == LEFT)
	{
		double *rightGhost = new double[blockDimY];	
		for(int j=0; j<blockDimY; ++j) 
			rightGhost[j] = temperature[index(blockDimX-1, j)];
		ghostProxy[pe].setBuffer(thisProxy,blockDimY,iter,thisIndex.x,thisIndex.y,LEFT,rightGhost);
		delete [] rightGhost;
	}
	
	if(dir == TOP)
	{
		double *bottomGhost =  new double[blockDimX];
		for(int i=0; i<blockDimX; ++i) 
			bottomGhost[i] = temperature[index(i, blockDimY-1)];
		ghostProxy[pe].setBuffer(thisProxy,blockDimX,iter,thisIndex.x,thisIndex.y,TOP,bottomGhost);
		delete [] bottomGhost;
	}

	if(dir == BOTTOM)
	{
		double *topGhost =  new double[blockDimX];
		for(int i=0; i<blockDimX; ++i) 
			topGhost[i] = temperature[index(i, 0)];
		ghostProxy[pe].setBuffer(thisProxy,blockDimX,iter,thisIndex.x,thisIndex.y,BOTTOM,topGhost);
		delete [] topGhost;
	}
  }

  void check_and_compute(CkCallback cb, int numSteps) {
    optimized_compute();
    contribute(sizeof(double), &maxdifference, CkReduction::max_double, cb);
  }


  void optimized_compute()
  {
    double temperatureIth=0.;
    double difference=0.;
    int x = thisIndex.x;
    int y = thisIndex.y;
    double *left,*right,*top,*bottom;
    double leftVal,rightVal,topVal,bottomVal;
    maxdifference=0.;

    if(istart == 0)
	    left = ghostProxy.ckLocalBranch()->getBuffer(LEFT,x-1,y);
    if(jstart == 0)
	    top = ghostProxy.ckLocalBranch()->getBuffer(TOP,x,y-1);
    if(ifinish==blockDimX)
	    right =ghostProxy.ckLocalBranch()->getBuffer(RIGHT,x+1,y);
    if(jfinish==blockDimY)
	    bottom = ghostProxy.ckLocalBranch()->getBuffer(BOTTOM,x,y+1);

#pragma unroll    
    for(int i=istart; i<ifinish; ++i) {
      for(int j=jstart; j<jfinish; ++j) {
	// calculate discrete mean value property 5 pt stencil
	if(i==0)
		leftVal = left[j];
	else
		leftVal   = temperature[index(i-1,j)];

	if(j==0)
		topVal = top[i];
	else
		topVal    = temperature[index(i,j-1)];

	if(i==blockDimX-1)
		rightVal = right[j];
	else
		rightVal  = temperature[index(i+1,j)];

	if(j==blockDimY-1)
		bottomVal = bottom[i];
	else
		bottomVal = temperature[index(i,j+1)];
		
	temperatureIth=(temperature[index(i, j)] 
			+leftVal + rightVal
			+topVal  + bottomVal) * 0.2;

	// update relative error
	difference=temperatureIth - temperature[index(i, j)];

	// fix sign without fabs overhead
	if(difference<0) difference*=-1.0; 
	maxdifference=(maxdifference>difference) ? maxdifference : difference;
	new_temperature[index(i, j)] = temperatureIth;
      }
    }
	if(istart==0)
		ghostProxy.ckLocalBranch()->deleteBuffer(LEFT,x-1,y);
	if(jstart==0)
		ghostProxy.ckLocalBranch()->deleteBuffer(TOP,x,y-1);
	if(ifinish==blockDimX)
		ghostProxy.ckLocalBranch()->deleteBuffer(RIGHT,x+1,y);
	if(jfinish==blockDimY)
		ghostProxy.ckLocalBranch()->deleteBuffer(BOTTOM,x,y+1);

  }

  // When all neighbor values have been received, 
  // we update our values and proceed
  // Enforce some boundary conditions
  void constrainBC() {
    if(topBound)
      for(int i=0; i<blockDimX; ++i)
	temperature[index(i, 0)] = 1.0;
    if(leftBound)
      for(int j=0; j<blockDimY; ++j)
	temperature[index(0, j)] = 1.0;

    if(bottomBound)
      for(int i=0; i<blockDimX; ++i)
	temperature[index(i, blockDimY-1)] = 0.;
    if(rightBound)
      for(int j=0; j<blockDimY; ++j)
	temperature[index(blockDimX-1, j)] = 0.;
  }
  // for debugging
 void dumpMatrix(double *matrix)
  {
    CkPrintf("[%d,%d]\n",thisIndex.x, thisIndex.y);
    for(int i=0; i<blockDimX;++i)
      {
	for(int j=0; j<blockDimY;++j)
	  {
	    CkPrintf("%0.3lf ",matrix[index(i,j)]);
	  }
	CkPrintf("\n");
      }
  }
};


#include "jacobi2d.def.h"
