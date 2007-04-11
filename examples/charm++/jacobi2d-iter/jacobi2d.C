#include "liveViz.h"
#include "jacobi2d.decl.h"

// See README for documentation

/*readonly*/ CProxy_Main mainProxy;

/*readonly*/ int num_rows;
/*readonly*/ int num_cols;

CkArrayID a;

#define total_iterations 128

class Main : public CBase_Main
{
public:
    int recieve_count;
    CProxy_Jacobi array;
    int num_chares;
    int iterations;

    Main(CkArgMsg* m) {
        // set iteration counter to zero
        iterations=0;

        // store the main proxy
        mainProxy = thisProxy;

        // specify the number of worker chares in each dimension
        num_rows = num_cols = 64;

        // print info
        CkPrintf("Running Jacobi on %d processors with (%d,%d) elements\n",
            CkNumPes(), num_rows, num_cols);

        // Create new array of worker chares
        array = CProxy_Jacobi::ckNew(num_rows, num_cols);

        // save the total number of worker chares we have in this simulation
        num_chares = num_rows*num_cols;

        // setup liveviz
        CkCallback c(CkIndex_Jacobi::requestNextFrame(0),array);
        liveVizConfig cfg(liveVizConfig::pix_color,false);
        liveVizInit(cfg,a,c);

        //Start the computation
        recieve_count = 0;
        array.begin_iteration();
    }

    // Each worker reports back to here when it completes an iteration
    void report(int row, int col, float value) {
        recieve_count++;
        if (num_chares == recieve_count) {
            if (iterations == total_iterations) {
                CkPrintf("Completed %d iterations\n", iterations);
                CkExit();
            } else {
                CkPrintf("starting new iteration.\n");
                recieve_count=0;
                iterations++;
                // Call begin_iteration on all worker chares in array
                array.begin_iteration();
            }
        }
    }
};

class Jacobi: public CBase_Jacobi {
public:
    double temperature;

    double update;
    int messages_due;

    // Constructor, initialize values
    Jacobi() {
       temperature = 1.0;
    }

    // a necessary function which we ignore now
    // if we were to use load balancing and migration
    // this function might become useful
    Jacobi(CkMigrateMessage* m) {}


    // Perform one iteration of work
    // The first step is to send the local state to the neighbors
    void begin_iteration(void) {
        messages_due = 4;
        update = 0.0;

        // Nodes on an edge shouldn't send messages to non-existant chares.
        if (thisIndex.x == 0)
            messages_due--;
        else
            thisProxy(thisIndex.x-1, thisIndex.y).recieve_neighbor(thisIndex.x, thisIndex.y, temperature);


        if (thisIndex.x == num_rows-1)
            messages_due--;
        else
            thisProxy(thisIndex.x+1, thisIndex.y).recieve_neighbor(thisIndex.x, thisIndex.y, temperature);


        if (thisIndex.y == 0)
            messages_due--;
        else
            thisProxy(thisIndex.x, thisIndex.y-1).recieve_neighbor(thisIndex.x, thisIndex.y, temperature);


        if (thisIndex.y == num_cols-1)
            messages_due--;
        else
            thisProxy(thisIndex.x, thisIndex.y+1).recieve_neighbor(thisIndex.x, thisIndex.y, temperature);


        check_done_iteration();
    }

    void recieve_neighbor(int neighbor_x, int neighbor_y, float t) {
        // we have just received a message from worker neighbor_x,neighbor_y with
        // a its current temperature. This worker's index is thisIndex.x, thisIndex.y
        update += t;
        messages_due--;
        check_done_iteration();
    }


    // check to see if we have received all neighbor values yet
    void check_done_iteration() {
        if (messages_due == 0) {
            temperature = (update+temperature) / 5.0;
            mainProxy.report(thisIndex.x, thisIndex.y, 1.0);
        }
    }


    // provide my portion of the image to the graphical liveViz client
    // Currently we just provide some pretty color depending upon the thread id
    // In a real program we would provide a colored rectangle or pixel that
    // depends upon the local thread data.
    void requestNextFrame(liveVizRequestMsg *m){
        int wdes = m->req.wid;
        int hdes = m->req.ht;

        CkPrintf("%d,%d requestNextFrame() with desired size %dx%d\n", thisIndex.x, thisIndex.y, wdes, hdes);

        // I will just deposit a single colored pixel to liveviz
        // Normally you would deposit some rectangular region

        // where to deposit
        int sx=thisIndex.x;
        int sy=thisIndex.y;
        int w=1,h=1; // Size of my rectangular part of the image

        // set the output pixel values for my rectangle
        // Each component is a char which can have 256 possible values.
        unsigned char *intensity= new unsigned char[3*w*h];
        intensity[0] = (4*(256+thisIndex.x-thisIndex.y)) % 256; // RED component
        intensity[1] = (4*(thisIndex.x+thisIndex.y)) % 256; // BLUE component
        intensity[2] = (4*thisIndex.y) % 256; // GREEN component

        liveVizDeposit(m, sx,sy, w,h, intensity, this);
        delete[] intensity;

    }



};

#include "jacobi2d.def.h"
