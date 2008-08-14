#include "LB_Bench.decl.h"
#include "charm++.h"

// See README for documentation

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int num_chare_rows;
/*readonly*/ int num_chare_cols;

// We want to wrap entries around, and because mod operator % sometimes misbehaves on negative values, 
// I just wrote these simple wrappers that will make the mod work as expected. -1 maps to the highest value.
#define wrap_x(a)  (((a)+num_chare_cols)%num_chare_cols)
#define wrap_y(a)  (((a)+num_chare_rows)%num_chare_rows)

CkArrayID a;

#define total_iterations 200

class Main : public CBase_Main
{
public:
  int done_count;
  CProxy_LB_Bench array;
  int num_chares;

  Main(CkArgMsg* m) {
	if (m->argc < 2) {
	  CkPrintf("%s [number chares per dimension]\n", m->argv[0]);
	  CkAbort("Abort");
	}

	// store the main proxy
	mainProxy = thisProxy;

	num_chare_rows = atoi(m->argv[1]);
	num_chare_cols = atoi(m->argv[1]);

	// print info
	CkPrintf("Running on %d processors with a %d x %d chare array\n", CkNumPes(), num_chare_rows, num_chare_cols);

	// Create new array of worker chares
	array = CProxy_LB_Bench::ckNew(num_chare_cols, num_chare_rows);

	// save the total number of worker chares we have in this simulation
	num_chares = num_chare_rows*num_chare_cols;

	//Start the computation
	done_count = 0;
	array.startup();
  }

  // Each worker reports back to here when it completes an iteration
  void report_done() {
	done_count++;
	if (num_chares == done_count) {
	  CkPrintf("Done\n");
	  CkExit();
	}
  }

};

class LB_Bench: public CBase_LB_Bench {
public:
  int iterations;
  int received_right, received_left, received_up, received_down;

  // Constructor
  LB_Bench() : iterations(0), received_right(0), received_down(0), received_up(0), received_left(0) {
	/*	iterations=0;
	  received_right=0;
	  received_left=0;
	  received_up=0;
	  received_down=0;
	*/
  }

  // For migration
  LB_Bench(CkMigrateMessage* m) {}

  // Destructor
  ~LB_Bench() { }

  // Perform one iteration of work
  // The first step is to send the local state to the neighbors
  void startup(void) {
	next_iter();
  }

  void next_iter(){
	thisProxy(wrap_x(thisIndex.x-1), thisIndex.y).fromRight();
	thisProxy(wrap_x(thisIndex.x+1), thisIndex.y).fromLeft();
	thisProxy(thisIndex.x, wrap_y(thisIndex.y-1)).fromDown();
	thisProxy(thisIndex.x, wrap_y(thisIndex.y+1)).fromUp();

	if(iterations % 10 ==5 &&  usesAtSync==CmiTrue )
	  AtSync();


  }

  void fromRight() {
	received_right ++;
	//	CkPrintf("%d,%d R=%d L=%d D=%d U=%d\n", thisIndex.x, thisIndex.y, received_right, received_left, received_down, received_up);
	//	CkAssert(received_right <= 2);
	check_and_compute();
  }

  void fromLeft() {
	received_left ++;
	//	CkPrintf("%d,%d R=%d L=%d D=%d U=%d\n", thisIndex.x, thisIndex.y, received_right, received_left, received_down, received_up);
	//	CkAssert(received_left <= 2);
	check_and_compute();
  }

  void fromDown() {
	received_down ++;
	//	CkPrintf("%d,%d R=%d L=%d D=%d U=%d\n", thisIndex.x, thisIndex.y, received_right, received_left, received_down, received_up);
	//CkAssert(received_down <= 2);
	check_and_compute();
  }

  void fromUp() {
	received_up ++;
	//	CkPrintf("%d,%d R=%d L=%d D=%d U=%d\n", thisIndex.x, thisIndex.y, received_right, received_left, received_down, received_up);
	//	CkAssert(received_up <= 2);
	check_and_compute();
  }

  void check_and_compute() {
	if(received_right>0 && received_left>0 && received_up>0 && received_down>0) {
	  //CkPrintf("%d,%d ====================== \n", thisIndex.x, thisIndex.y);

	  received_right --;
	  received_left --;
	  received_down --;
	  received_up --;
	  
	  //	  CkPrintf("%d,%d R=%d L=%d D=%d U=%d\n", thisIndex.x, thisIndex.y, received_right, received_left, received_down, received_up);

	  if(iterations < total_iterations){
		iterations++;
		thisProxy(thisIndex.x,thisIndex.y).compute();
	  } else {
		mainProxy.report_done();
	  }

	}
  }


  void compute() {
	double work_factor = 1.0;

	//	CkPrintf("my x index is %d of %d, iteration=%d\n", thisIndex.x, num_chare_cols, iterations);
	if(thisIndex.x >= num_chare_cols*0.40 && thisIndex.x <= num_chare_cols*0.60){
	  const double start_activate=0.4*(double)total_iterations;
	  const double end_activate=0.7*(double)total_iterations;
	  double fraction_activated;

	  if(iterations < start_activate)
		fraction_activated = 0.0;
	  else if(iterations > end_activate)
		fraction_activated = 1.0;
	  else
		fraction_activated = ((double)iterations-start_activate) / (end_activate-start_activate); 

	  if( ((double)thisIndex.y / ((double)num_chare_rows-1.0)) <= fraction_activated)
		work_factor += num_chare_rows*2.0;
	  //	  CkPrintf("x index %d has work_factor %f at iteration %d\n", thisIndex.x, work_factor, iterations);
	}

	double a[2000], b[2000], c[2000];
	for(int j=0;j<100*work_factor;j++){
	  for(int i=0;i<2000;i++){
		a[i] = 7.0;
		b[i] = 5.0;
	  }
	  for(int i=0;i<2000/2;i++){
		c[i] = a[2*i]*b[2*i+1]*a[i];
		c[2*i] = a[2*i];
	  }
	}

	next_iter();
  }
 


  void ResumeFromSync(void) { //Called by Load-balancing framework
	// CkPrintf("Element %d,%d resumeFromSync on PE %d\n",thisIndex.x, thisIndex.y, CkMyPe());
  }


  virtual void pup(PUP::er &p)
  {
    CBase_LB_Bench::pup(p);
    p | iterations;
    p | received_right;
    p | received_left;
    p | received_up;
    p | received_down;
  }




};

#include "LB_Bench.def.h"
