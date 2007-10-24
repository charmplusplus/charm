/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/** \file jacobi2d.C
 *  Author: Abhinav S Bhatele
 *  Date Created: October 24th, 2007
 *
 *  This does a topological placement for a 2d jacobi.
 *  This jacobi is different from the one in ../../jacobi2d-iter in
 *  the sense that it does not use barriers
 */

#include "jacobi2d.decl.h"

// See README for documentation

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int block_height;
/*readonly*/ int block_width;
/*readonly*/ int array_height;
/*readonly*/ int array_width;

// specify the number of worker chares in each dimension
/*readonly*/ int num_chare_rows;
/*readonly*/ int num_chare_cols;

// We want to wrap entries around, and because mod operator % 
// sometimes misbehaves on negative values. -1 maps to the highest value.
#define wrap_x(a)  (((a)+num_chare_cols)%num_chare_cols)
#define wrap_y(a)  (((a)+num_chare_rows)%num_chare_rows)

CkArrayID a;

#define MAX_ITER	200
#define LEFT		1
#define RIGHT		2
#define TOP		3
#define BOTTOM		4

class Main : public CBase_Main
{
  public:
    int recieve_count;
    CProxy_Jacobi array;
    int num_chares;
    int iterations;

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

        // Create new array of worker chares
        array = CProxy_Jacobi::ckNew(num_chare_cols, num_chare_rows);

        // save the total number of worker chares we have in this simulation
        num_chares = num_chare_rows*num_chare_cols;

        //Start the computation
        recieve_count = 0;
        array.begin_iteration();
    }

    // Each worker reports back to here when it completes an iteration
    void report(int row, int col) {
        recieve_count++;
        //CkPrintf("ROW %d %d\n", row, col);
	//CkPrintf("%d %d\n", recieve_count, num_chares);
        if (num_chares == recieve_count) {
          CkExit();
        }
    }
};

class Jacobi: public CBase_Jacobi {
  public:
    int arrived_left;
    int arrived_right;
    int arrived_top;
    int arrived_bottom;
    int iterations;
    int msgs;

    double **temperature;

    // Constructor, initialize values
    Jacobi() {
        int i,j;
        // allocate two dimensional array
        temperature = new double*[block_height+2];
        for (i=0; i<block_height+2; i++)
          temperature[i] = new double[block_width+2];
        for(i=0;i<block_height+2;++i){
            for(j=0;j<block_width+2;++j){
                temperature[i][j] = 0.0;
            }
        }
	iterations = 0;
	arrived_left = 0;
	arrived_right = 0;
	arrived_top = 0;
	arrived_bottom = 0;
	msgs = 0;
        BC();
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

    void ghostsFromRight(int width, double ghost_values[]) {
        for(int i=0;i<width;++i){
            temperature[i+1][block_width+1] = ghost_values[i];
        }
        check_and_compute(RIGHT);
    }

    void ghostsFromLeft(int width, double ghost_values[]) {
        for(int i=0;i<width;++i){
            temperature[i+1][0] = ghost_values[i];
        }
        check_and_compute(LEFT);
    }

    void ghostsFromBottom(int width, double ghost_values[]) {
        for(int i=0;i<width;++i){
            temperature[block_height+1][i+1] = ghost_values[i];
        }
        check_and_compute(BOTTOM);
    }

    void ghostsFromTop(int width, double ghost_values[]) {
        for(int i=0;i<width;++i){
            temperature[0][i+1] = ghost_values[i];
        }
        check_and_compute(TOP);
    }

    void check_and_compute(int direction) {
        switch(direction) {
	  case LEFT:
	    arrived_left++;
	    msgs++;
	    break;
	  case RIGHT:
	    arrived_right++;
	    msgs++;
	    break;
	  case TOP:
	    arrived_top++;
	    msgs++;
	    break;
	  case BOTTOM:
	    arrived_bottom++;
	    msgs++;
	    break;
	}
        if (arrived_left >=1 && arrived_right >=1 && arrived_top >=1 && arrived_bottom >=1) {
	  arrived_left--;
	  arrived_right--;
	  arrived_top--;
	  arrived_bottom--;
          compute();
	  iterations++;
          if (iterations == MAX_ITER) {
            if(thisIndex.x==0 && thisIndex.y==0) CkPrintf("Completed %d iterations\n", iterations);
            //CkPrintf("INDEX %d %d MSGS %d\n", thisIndex.x, thisIndex.y, msgs);
            mainProxy.report(thisIndex.x, thisIndex.y);
            // CkExit();
          } else {
            if(thisIndex.x==0 && thisIndex.y==0) CkPrintf("Starting new iteration %d.\n", iterations);
            // Call begin_iteration on all worker chares in array
            begin_iteration();
          } 
        }
    }

    // Check to see if we have received all neighbor values yet
    // If all neighbor values have been received, we update our values and proceed
    void compute() {
        // We must create a new array for these values because we don't want to 
        // update any of the values in temperature[][] array until using them first.
        // Other schemes could be used to accomplish this same problem. We just put
        // the new values in a temporary array and write them to temperature[][] 
        // after all of the new values are computed.
        double new_temperature[block_height+2][block_width+2];
    
        for(int i=1;i<block_height+1;++i) {
            for(int j=1;j<block_width+1;++j) {
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
