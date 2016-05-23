#include "jacobi2d.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int block_height;
/*readonly*/ int block_width;
/*readonly*/ int array_height;
/*readonly*/ int array_width;

// specify the number of worker chares in each dimension
/*readonly*/ int num_chare_rows;
/*readonly*/ int num_chare_cols;

// We want to wrap entries around, and because mod operator % sometimes misbehaves on negative values,
// I just wrote these simple wrappers that will make the mod work as expected. -1 maps to the highest value.
#define wrap_x(a)  (((a)+num_chare_cols)%num_chare_cols)
#define wrap_y(a)  (((a)+num_chare_rows)%num_chare_rows)


class Main : public CBase_Main
{
public:
    int recieve_count;
    CProxy_Jacobi array;
    int num_chares;
    int iterations;
    int total_iterations;
	double stTime;
    double startTime;

    Main(CkArgMsg* m) {
        if (m->argc < 3) {
          CkPrintf("%s [array_size] [block_size]\n", m->argv[0]);
          CkAbort("Abort");
        }

        // set iteration counter to zero
        iterations=0;

        // store the main proxy
        mainProxy = thisProxy;

        array_height = array_width = atoi(m->argv[1]);
        block_height = block_width = atoi(m->argv[2]);
        if (array_width < block_width || array_width % block_width != 0)
          CkAbort("array_size % block_size != 0!");

        num_chare_rows = array_height / block_height;
        num_chare_cols = array_width / block_width;
        // print info
        CkPrintf("Running Jacobi on %d processors with (%d,%d) elements\n", CkNumPes(), num_chare_rows, num_chare_cols);

		stTime = CkWallTimer();
	    total_iterations = 400;
	    if (m->argc > 3) {
	        total_iterations = atoi(m->argv[3]);
	    }

        // Create new array of worker chares
        array = CProxy_Jacobi::ckNew(num_chare_cols, num_chare_rows);

        // save the total number of worker chares we have in this simulation
        num_chares = num_chare_rows*num_chare_cols;

        //Start the computation
        startTime = CkWallTimer();
        recieve_count = 0;
        array.begin_iteration();
    }

  Main(CkMigrateMessage *m) : CBase_Main(m) {
    if (m!=NULL) {
      CkArgMsg *args = (CkArgMsg *)m;
      CkPrintf("Received %d arguments: { ",args->argc);
      for (int i=0; i<args->argc; ++i) {
        CkPrintf("|%s| ",args->argv[i]);
      }
      CkPrintf("}\n");
    } else {
      CkPrintf("Arguments null\n");
    }
      // subtle: Chare proxy readonly needs to be updated manually because of
      // the object pointer inside it.
    mainProxy = thisProxy;

    CkPrintf("Resuming Jacobi on %d processors with (%d,%d) elements\n", CkNumPes(), num_chare_rows, num_chare_cols);

    CkPrintf("Main's MigCtor.\n");
  }

    // Each worker reports back to here when it completes an iteration
void report(CkReductionMsg *m) {
    if (m==NULL) CkAbort("Null Red msg\n");
    iterations=*((int *)m->getData());
    recieve_count++;
    double totaltime = CkWallTimer() - startTime;
	if (1 == recieve_count) {
        if (iterations == total_iterations || CkWallTimer()-stTime>=3000000) {
			CkPrintf("Program Done! avg_it:%.6f\n",(CkWallTimer()-stTime)/iterations);
            CkExit();
        } else {
            if(iterations%1==0) CkPrintf("starting new iteration; iteration %d time: %.6lf time/itr::%.6f\n", iterations, CkWallTimer()-stTime,(CkWallTimer()-stTime)/iterations);
            CkPrintf("Memory Usage: %d bytes \n", CmiMemoryUsage());
            recieve_count=0;
            iterations++;
            // Call begin_iteration on all worker chares in array
            startTime = CkWallTimer();
            array.begin_iteration();
        }
    }
}
void pup(PUP::er &p){
    p|recieve_count;
    p|array;
    p|num_chares;
    p|iterations;
    p|total_iterations;
    p|stTime;
    p|startTime;
    CkPrintf("Main's PUPer. \n");
  }


};

class Jacobi: public CBase_Jacobi {
public:
    int messages_due;
	int iteration;
    int useLB;
    double **temperature;

    // Constructor, initialize values
    Jacobi() {
        int i,j;
	    iteration = 0;
        useLB = 1;
        usesAtSync = true;

        // allocate two dimensional array
        temperature = new double*[block_height+2];
        for (i=0; i<block_height+2; i++)
            temperature[i] = new double[block_width+2];
        messages_due = 4;
        for(i=0;i<block_height+2;++i){
            for(j=0;j<block_width+2;++j){
                temperature[i][j] = 0.0;
            }
        }
        BC();
    }

    void pup(PUP::er &p){
        p|messages_due;
        p|iteration;
        p|useLB;
        if (p.isUnpacking()) temperature=new double*[block_height+2];
        for (int i=0;i<block_height+2;i++) {
                if (p.isUnpacking()) temperature[i]=new double[block_width+2]; // allocate i’th foo
                p(temperature[i],block_height+2); //pup the i’th foo
        }
        /* There may be some more variables used in doWork */
    }


    // Enforce some boundary conditions
    void BC(){
        // Heat left and top edges of each chare's block
	    for(int i=1;i<block_height+1;++i)
            temperature[i][1] = 255.0;
        for(int j=1;j<block_width+1;++j)
            temperature[1][j] = 255.0;
    }

    // a necessary function which we ignore now
    // if we were to use load balancing and migration
    // this function might become useful
    Jacobi(CkMigrateMessage* m) {}

    ~Jacobi() {
      for (int i=0; i<block_height; i++)
        delete [] temperature[i];
      delete [] temperature;
    }

    // Perform one iteration of work
    // The first step is to send the local state to the neighbors
    void begin_iteration(void) {
        if (iteration %50 ==0 && useLB ) {
            useLB = 0;
            if(thisIndex.x==0 && thisIndex.y==0) CkPrintf("PROC#%d Calling LBD --------------------- iteration=%d\n",CkMyPe(),iteration);
            AtSync();
        } else {

        useLB=1;
        if(thisIndex.x==0 && thisIndex.y==0) CkPrintf("PROC#%d started --------------------- iteration=%d\n",CkMyPe(),iteration);
				iteration++;
        // Copy left column and right column into temporary arrays
        double *left_edge = new double[block_height];
        double *right_edge = new double[block_height];

        for(int i=0;i<block_height;++i){
            left_edge[i] = temperature[i+1][1];
            right_edge[i] = temperature[i+1][block_width];
        }

        // Send my left edge
        thisProxy(wrap_x(thisIndex.x-1), thisIndex.y).ghostsFromRight(block_height, left_edge);
		// Send my right edge
        thisProxy(wrap_x(thisIndex.x+1), thisIndex.y).ghostsFromLeft(block_height, right_edge);
		// Send my top edge
        thisProxy(thisIndex.x, wrap_y(thisIndex.y-1)).ghostsFromBottom(block_width, &temperature[1][1]);
		// Send my bottom edge
        thisProxy(thisIndex.x, wrap_y(thisIndex.y+1)).ghostsFromTop(block_width, &temperature[block_height][1]);

        delete [] right_edge;
        delete [] left_edge;
    }
}

void ResumeFromSync() {begin_iteration();}

    void ghostsFromRight(int width, double ghost_values[]) {
        for(int i=0;i<width;++i){
            temperature[i+1][block_width+1] = ghost_values[i];
        }
        check_and_compute();
    }

    void ghostsFromLeft(int width, double ghost_values[]) {
        for(int i=0;i<width;++i){
            temperature[i+1][0] = ghost_values[i];
        }
        check_and_compute();
    }

    void ghostsFromBottom(int width, double ghost_values[]) {
        for(int i=0;i<width;++i){
            temperature[block_height+1][i+1] = ghost_values[i];
        }
        check_and_compute();
    }

    void ghostsFromTop(int width, double ghost_values[]) {
        for(int i=0;i<width;++i){
            temperature[0][i+1] = ghost_values[i];
        }
        check_and_compute();
    }

    void check_and_compute() {
       if (--messages_due == 0) {
            messages_due = 4;
            compute();
		    contribute(sizeof(int),&iteration,CkReduction::max_int,CkCallback(CkIndex_Main::report(NULL),mainProxy));
        }
    }

    // Check to see if we have received all neighbor values yet
    // If all neighbor values have been received, we update our values and proceed
    void compute() {
            // We must create a new array for these values because we don't want to update any of the
            // the values in temperature[][] array until using them first. Other schemes could be used
            // to accomplish this same problem. We just put the new values in a temporary array
            // and write them to temperature[][] after all of the new values are computed.
            double new_temperature[block_height+2][block_width+2];

            for(int i=1;i<block_height+1;++i){
                for(int j=1;j<block_width+1;++j){
                    // update my value based on the surrounding values
                    new_temperature[i][j] = (temperature[i-1][j]+temperature[i+1][j]+temperature[i][j-1]+temperature[i][j+1]+temperature[i][j]) / 5.0;

                }
            }

            for(int i=0;i<block_height+2;++i)
                for(int j=0;j<block_width+2;++j)
                    temperature[i][j] = new_temperature[i][j];

            // Enforce the boundary conditions again
            BC();

    }
};

#include "jacobi2d.def.h"
