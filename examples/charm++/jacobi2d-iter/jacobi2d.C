#include "liveViz.h"
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

// We want to wrap entries around, and because mod operator % sometimes misbehaves on negative values, 
// I just wrote these simple wrappers that will make the mod work as expected. -1 maps to the highest value.
#define wrap_x(a)  (((a)+num_chare_cols)%num_chare_cols)
#define wrap_y(a)  (((a)+num_chare_rows)%num_chare_rows)

CkArrayID a;

//#define total_iterations 200

class Main : public CBase_Main
{
public:
    int recieve_count;
    CProxy_Jacobi array;
    int num_chares;
    int iterations;
    int total_iterations;
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

	total_iterations = 200;
	if (m->argc > 3) {
	  total_iterations = atoi(m->argv[3]);
	}

        // Create new array of worker chares
        array = CProxy_Jacobi::ckNew(num_chare_cols, num_chare_rows);

        // save the total number of worker chares we have in this simulation
        num_chares = num_chare_rows*num_chare_cols;

        // setup liveviz
        CkCallback c(CkIndex_Jacobi::requestNextFrame(0),array);
        liveVizConfig cfg(liveVizConfig::pix_color,true);
        liveVizInit(cfg,a,c);

        //Start the computation
        startTime = CkWallTimer();
        recieve_count = 0;
        array.begin_iteration();
    }

    // Each worker reports back to here when it completes an iteration
    void report(int row, int col) {
        recieve_count++;
        double totaltime = CkWallTimer() - startTime;
        if (num_chares == recieve_count) {
            if (iterations == total_iterations) {
                CkPrintf("Completed %d iterations; last iteration time: %.6lf\n", iterations, totaltime);
                CkExit();
            } else {
                CkPrintf("starting new iteration; iteration %d time: %.6lf\n", iterations, totaltime);
                recieve_count=0;
                iterations++;
                // Call begin_iteration on all worker chares in array
                startTime = CkWallTimer();
                array.begin_iteration();
            }
        }
    }
};

class Jacobi: public CBase_Jacobi {
public:
    int messages_due;

    double **temperature;

    // Constructor, initialize values
    Jacobi() {
        int i,j;
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
          mainProxy.report(thisIndex.x, thisIndex.y);
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


    // provide my portion of the image to the graphical liveViz client
    // Currently we just provide some pretty color depending upon the thread id
    // In a real program we would provide a colored rectangle or pixel that
    // depends upon the local thread data.
    void requestNextFrame(liveVizRequestMsg *m){
		// These specify the desired total image size requested by the client viewer
        int wdes = m->req.wid;
        int hdes = m->req.ht;

        // Deposit a rectangular region to liveViz

        // where to deposit
        int sx=thisIndex.x*block_width;
        int sy=thisIndex.y*block_height;
        int w=block_width,h=block_height; // Size of my rectangular part of the image

        // set the output pixel values for my rectangle
        // Each component is a char which can have 256 possible values.
        unsigned char *intensity= new unsigned char[3*w*h];
        for(int i=0;i<h;++i){
            for(int j=0;j<w;++j){
                        intensity[3*(i*w+j)+0] = 255; // RED component
                        intensity[3*(i*w+j)+1] = 255-temperature[i+1][j+1]; // BLUE component
                        intensity[3*(i*w+j)+2] = 255-temperature[i+1][j+1]; // GREEN component
            }
        }

        liveVizDeposit(m, sx,sy, w,h, intensity, this);
        delete[] intensity;

    }



};

#include "jacobi2d.def.h"
