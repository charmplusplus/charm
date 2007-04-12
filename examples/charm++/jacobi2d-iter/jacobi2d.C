#include "liveViz.h"
#include "jacobi2d.decl.h"

// See README for documentation

/*readonly*/ CProxy_Main mainProxy;

// specify the number of worker chares in each dimension
#define num_chare_rows 5
#define num_chare_cols 3

// Each worker chare will process a 4x4 block of elements
#define block_width 131
#define block_height 61

// We want to wrap entries around, and because mod operator % sometimes misbehaves on negative values, 
// I just wrote these simple wrappers that will make the mod work as expected. -1 maps to the highest value.
#define wrap_x(a)  (((a)+num_chare_cols)%num_chare_cols)
#define wrap_y(a)  (((a)+num_chare_rows)%num_chare_rows)

CkArrayID a;

#define total_iterations 200

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

        // print info
        CkPrintf("Running Jacobi on %d processors with (%d,%d) elements\n", CkNumPes(), num_chare_rows, num_chare_cols);

        // Create new array of worker chares
        array = CProxy_Jacobi::ckNew(num_chare_cols, num_chare_rows);

        // save the total number of worker chares we have in this simulation
        num_chares = num_chare_rows*num_chare_cols;

        // setup liveviz
        CkCallback c(CkIndex_Jacobi::requestNextFrame(0),array);
        liveVizConfig cfg(liveVizConfig::pix_color,true);
        liveVizInit(cfg,a,c);

        //Start the computation
        recieve_count = 0;
        array.begin_iteration();
    }

    // Each worker reports back to here when it completes an iteration
    void report(int row, int col) {
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
    int messages_due;

    double temperature[block_height][block_width];
    double left_ghosts[block_height];
    double right_ghosts[block_height];
    double top_ghosts[block_width];
    double bottom_ghosts[block_width];

    // Constructor, initialize values
    Jacobi() {
        for(int i=0;i<block_height;++i){
            for(int j=0;j<block_width;++j){
                temperature[i][j] = 0.0;
            }
        }
        BC();
    }

    // Enforce some boundary conditions
    void BC(){
        // Heat left and top edges of each chare's block
		for(int i=0;i<block_height;++i)
            temperature[i][0] = 255.0;
        for(int j=0;j<block_width;++j)
            temperature[0][j] = 255.0;
    }

    // a necessary function which we ignore now
    // if we were to use load balancing and migration
    // this function might become useful
    Jacobi(CkMigrateMessage* m) {}


    // Perform one iteration of work
    // The first step is to send the local state to the neighbors
    void begin_iteration(void) {
        messages_due = 4;

        // Copy left column and right column into temporary arrays
        double left_edge[block_height];
        double right_edge[block_height];
		double top_edge[block_width];
		double bottom_edge[block_height];

        for(int i=0;i<block_height;++i){
            left_edge[i] = temperature[i][0];
            right_edge[i] = temperature[i][block_width-1];
        }

        for(int j=0;j<block_width;++j){
            top_edge[j] = temperature[0][j];
            bottom_edge[j] = temperature[block_height-1][j];
        }

        // Send my left edge
        thisProxy(wrap_x(thisIndex.x-1), thisIndex.y).recieve_neighbor(thisIndex.x, thisIndex.y, block_height, left_edge);
		// Send my right edge
        thisProxy(wrap_x(thisIndex.x+1), thisIndex.y).recieve_neighbor(thisIndex.x, thisIndex.y, block_height, right_edge);
		// Send my top edge
        thisProxy(thisIndex.x, wrap_y(thisIndex.y-1)).recieve_neighbor(thisIndex.x, thisIndex.y, block_width, top_edge);
		// Send my bottom edge
        thisProxy(thisIndex.x, wrap_y(thisIndex.y+1)).recieve_neighbor(thisIndex.x, thisIndex.y, block_width, bottom_edge);

        check_done_iteration();
    }

    void recieve_neighbor(int neighbor_x, int neighbor_y, int width, double ghost_values[]) {
        // we have just received a message from worker neighbor_x,neighbor_y with its adjacent
        // row or column of data values. This worker's index is thisIndex.x, thisIndex.y
        // We store these in temporary arrays, until all data arrives, then we perform computation
        // This could be optimized by performing the available computation as soon as the 
        // required data arrives, but  this example is intentionally simple
        if(neighbor_x == wrap_x(thisIndex.x-1) && neighbor_y == thisIndex.y){
            // the ghost data from my LEFT neighbor
            CkAssert(width == block_height);
            for(int i=0;i<width;++i){
                left_ghosts[i] = ghost_values[i];
            }
        } else if(neighbor_x == wrap_x(thisIndex.x+1) && neighbor_y == thisIndex.y){
            // the ghost data from my RIGHT neighbor
            CkAssert(width == block_height);
            for(int i=0;i<width;++i){
                right_ghosts[i] = ghost_values[i];
            }
        } else  if(neighbor_x == thisIndex.x && neighbor_y == wrap_y(thisIndex.y-1)){
            // the ghost data from my TOP neighbor
            CkAssert(width == block_width);
            for(int i=0;i<width;++i){
                top_ghosts[i] = ghost_values[i];
            }
        } else if(neighbor_x == thisIndex.x && neighbor_y == wrap_y(thisIndex.y+1)){
            // the ghost data from my BOTTOM neighbor
            CkAssert(width == block_width);
            for(int i=0;i<width;++i){
                bottom_ghosts[i] = ghost_values[i];
            }
        } else {
            CkPrintf("Message from non-neighbor chare. I am %d,%d. Message was from %d,%d\n",thisIndex.x,thisIndex.y,neighbor_x,neighbor_y);
            CkExit();
        }

        messages_due--;
        check_done_iteration();
    }


    // Check to see if we have received all neighbor values yet
    // If all neighbor values have been received, we update our values and proceed
    void check_done_iteration() {
        if (messages_due == 0) {
            // We must create a new array for these values because we don't want to update any of the
            // the values in temperature[][] array until using them first. Other schemes could be used
            // to accomplish this same problem. We just put the new values in a temporary array
            // and write them to temperature[][] after all of the new values are computed.
            double new_temperature[block_height][block_width];
    
            for(int i=0;i<block_height;++i){
                for(int j=0;j<block_width;++j){
                    // first we find the values around the i,j entry in this chare's local block                
                    double up, down, left, right;

                    if(i==0)
                        up = top_ghosts[j];
                    else
                        up = temperature[i-1][j];

                    if(i==block_height-1)
                        down = bottom_ghosts[j];
                    else
                        down = temperature[i+1][j];

                    if(j==0)
                        left = left_ghosts[i];
                    else
                        left = temperature[i][j-1];

                    if(j==block_width-1)
                        right = right_ghosts[i];
                    else
                        right = temperature[i][j+1];


                    // update my value based on the surrounding values
					new_temperature[i][j] = (up+down+left+right+temperature[i][j]) / 5.0;

                }
            }

            for(int i=0;i<block_height;++i)
                for(int j=0;j<block_width;++j)
                    temperature[i][j] = new_temperature[i][j];

            // Enforce the boundary conditions again
            BC();

            mainProxy.report(thisIndex.x, thisIndex.y);
        }
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
                        intensity[3*(i*w+j)+1] = 255-temperature[i][j]; // BLUE component
                        intensity[3*(i*w+j)+2] = 255-temperature[i][j]; // GREEN component
            }
        }

        liveVizDeposit(m, sx,sy, w,h, intensity, this);
        delete[] intensity;

    }



};

#include "jacobi2d.def.h"
