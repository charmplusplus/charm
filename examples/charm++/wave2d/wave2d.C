#include "liveViz.h"
#include "wave2d.decl.h"

// Author: Isaac Dooley 2008

// This program solves the 2-d wave equation over a grid, displaying pretty results through liveViz
// The program could be made more efficient, but is kept this way for simplicity.
// Migration is not supported yet. Please add the PUP function if you get a chance!

// This program is based on the description here:
// http://www.mtnmath.com/whatrh/node66.html
// "The wave equation is the universal equation of physics. It works for light, 
//  sound, waves on the surface of water and a great deal more"


/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_Wave arrayProxy;

#define TotalDataWidth  800
#define TotalDataHeight 800

#define chareArrayWidth  4
#define chareArrayHeight  4
#define num_chares ((chareArrayWidth)*(chareArrayHeight))

#define total_iterations 5000

#define numInitialPertubations 5


// A modulo operator that works for a==-1
#define wrap(a,b)  (((a)+b)%b)

class Main : public CBase_Main
{
public:
  int recieve_count;
  int iterations;

  Main(CkArgMsg* m) {

    // set iteration counter to zero
    iterations=0;

    // store the main proxy
    mainProxy = thisProxy;

    // print info
    CkPrintf("Running wave2d on %d processors\n", CkNumPes());

    // Create new array of worker chares
    arrayProxy = CProxy_Wave::ckNew(chareArrayWidth, chareArrayHeight);

    // setup liveviz
    CkCallback c(CkIndex_Wave::requestNextFrame(0),arrayProxy);
    liveVizConfig cfg(liveVizConfig::pix_color,true);

    CkArrayID a; // unused???
    liveVizInit(cfg,a,c);

    //Start the computation
    recieve_count = 0;
    arrayProxy.begin_iteration();
  }



  // Each worker reports back to here when it completes an iteration
  void report(int row, int col) {

    recieve_count++;

    if (num_chares == recieve_count) {

      if (iterations == total_iterations) {
	CkPrintf("Program Done!\n");
	CkExit();
      } else {

	// Start the next iteration
	recieve_count=0;
	iterations++;
	CkPrintf("Completed %d iterations\n", iterations);

	arrayProxy.begin_iteration();
			  
      }
    }
  }


};

class Wave: public CBase_Wave {
public:
  int messages_due;
  int mywidth;
  int myheight;

  double *pressure;
  double *pressure_old;

  double *buffer_left;
  double *buffer_right;
  double *buffer_up;
  double *buffer_down;


  // Constructor, initialize values
  Wave() {

    mywidth=TotalDataWidth / chareArrayWidth;
    myheight= TotalDataHeight / chareArrayHeight;

    pressure = new double[mywidth*myheight];
    pressure_old = new double[mywidth*myheight];

    buffer_left = new double[myheight];
    buffer_right = new double[myheight];
    buffer_up = new double[mywidth];
    buffer_down = new double[mywidth];

    messages_due = 4;

    InitialConditions();
  }


  // Setup some Initial pressure pertubations
  void InitialConditions(){
    srand(0);

    for(int i=0;i<myheight*mywidth;i++){
        pressure[i] = 0.0;
        pressure_old[i] = 0.0;
    }
    
    for(int s=0; s<numInitialPertubations; s++){    
      // Randomly place a circle within the 2-d data array (without wrapping around)
      int radius = 20+rand() % 30;
      int xcenter = radius + rand() % (TotalDataWidth - 2*radius);
      int ycenter = radius + rand() % (TotalDataHeight - 2*radius);
      for(int i=0;i<myheight;i++){
	for(int j=0; j<mywidth; j++){
	  int globalx = thisIndex.x*mywidth + j; // The coordinate in the global data array (not just in this chare's portion)
	  int globaly = thisIndex.y*myheight + i;
	  double distanceToCenter = sqrt((globalx-xcenter)*(globalx-xcenter) + (globaly-ycenter)*(globaly-ycenter));
	  if (distanceToCenter < radius) {
	    double rscaled = distanceToCenter / radius; // ranges from 0 to 1
	    double rscaled2 = rscaled*3.0*3.14159/2.0; // ranges from 0 to 3pi/2						
	    double t = 400.0 * cos(rscaled2) ; // Range not to exceed -400 to 400
	    pressure[i*mywidth+j] = t;
	    pressure_old[i*mywidth+j] = t;
	  }
	}						
      }
    }
  }



  Wave(CkMigrateMessage* m) {
    CkAbort("Migration of this class is not supported yet. Write PUP function if migration is used\n"); 
  }

  ~Wave() { 
    delete [] pressure; 
    delete [] pressure_old;
  }

  void begin_iteration(void) {

    double *left_edge = new double[myheight];
    double *right_edge = new double[myheight];		
    double *top_edge = new double[mywidth];
    double *bottom_edge = new double[mywidth];

    for(int i=0;i<myheight;++i){
      left_edge[i] = pressure[i*mywidth];
      right_edge[i] = pressure[i*mywidth + mywidth-1];
    }

    for(int i=0;i<mywidth;++i){
      top_edge[i] = pressure[i];
      bottom_edge[i] = pressure[(myheight-1)*mywidth + i];
    }

    // Send my left edge
    thisProxy(wrap(thisIndex.x-1,chareArrayWidth), thisIndex.y).ghostsFromRight(myheight, left_edge);
    // Send my right edge
    thisProxy(wrap(thisIndex.x+1,chareArrayWidth), thisIndex.y).ghostsFromLeft(myheight, right_edge);
    // Send my top edge
    thisProxy(thisIndex.x, wrap(thisIndex.y-1,chareArrayHeight)).ghostsFromBottom(mywidth, top_edge);
    // Send my bottom edge
    thisProxy(thisIndex.x, wrap(thisIndex.y+1,chareArrayHeight)).ghostsFromTop(mywidth, bottom_edge);

    delete [] right_edge;
    delete [] left_edge;
    delete [] top_edge;
    delete [] bottom_edge;
  }

  void ghostsFromRight(int width, double ghost_values[]) {
    for(int i=0;i<width;++i){
      buffer_right[i] = ghost_values[i];
    }
    check_and_compute();
  }

  void ghostsFromLeft(int width, double ghost_values[]) {
    for(int i=0;i<width;++i){
      buffer_left[i] = ghost_values[i];
    }
    check_and_compute();
  }

  void ghostsFromBottom(int width, double ghost_values[]) {
    for(int i=0;i<width;++i){
      buffer_down[i] = ghost_values[i];
    }
    check_and_compute();
  }

  void ghostsFromTop(int width, double ghost_values[]) {
    for(int i=0;i<width;++i){
      buffer_up[i] = ghost_values[i];
    }
    check_and_compute();
  }

  void check_and_compute() {
    if (--messages_due == 0) {

      // Compute the new values based on the current and previous step values

      double *pressure_new = new double[mywidth*myheight];

      for(int i=0;i<myheight;++i){
	for(int j=0;j<mywidth;++j){

	  // Current step's values for neighboring array elements
	  double left  = (j==0          ? buffer_left[i]  : pressure[i*mywidth+j-1] );
	  double right = (j==mywidth-1  ? buffer_right[i] : pressure[i*mywidth+j+1] );
	  double up    = (i==0          ? buffer_up[j]    : pressure[(i-1)*mywidth+j] );
	  double down  = (i==myheight-1 ? buffer_down[j]  : pressure[(i+1)*mywidth+j] );

	  // Current values for this array element
	  double curr = pressure[i*mywidth+j];

	  // Previous step's value for this array element
	  double old  = pressure_old[i*mywidth+j];

	  // Wave speed
	  double c = 0.4;

	  // Compute the new value
	  pressure_new[i*mywidth+j] = c*c*(left+right+up+down - 4.0*curr)-old+2.0*curr;

	  // Round any near-zero values to zero (avoid denorms)
	  //	  if(pressure_new[i*mywidth+j] < 0.0001 && pressure_new[i*mywidth+j] >  -0.0001)
	  //  pressure_new[i*mywidth+j] = 0.0;

	}
      }
		
      // Advance to next step by copying values to the arrays for the previous steps
      for(int i=0;i<myheight;++i){
	for(int j=0;j<mywidth;++j){
	  pressure_old[i*mywidth+j] = pressure[i*mywidth+j];
	  pressure[i*mywidth+j] = pressure_new[i*mywidth+j];
	}
      }

      delete[] pressure_new;

      messages_due = 4;
      mainProxy.report(thisIndex.x, thisIndex.y);
    }
  }



  // provide my portion of the image to the graphical liveViz client                           
  void requestNextFrame(liveVizRequestMsg *m){
    // Draw my part of the image, plus a nice 1px border along my right/bottom boundary
    int sx=thisIndex.x*mywidth; // where to deposit
    int sy=thisIndex.y*myheight;
    int w=mywidth; // Size of my rectangular part of the image
    int h=myheight;
    
    // set the output pixel values for my rectangle
    // Each component is a char which can have 256 possible values.
    unsigned char *intensity= new unsigned char[3*w*h];
    for(int i=0;i<myheight;++i){
      for(int j=0;j<mywidth;++j){
        double t = pressure[i*mywidth+j];
        if(t > 255.0){
          t = 255.0;
        } else if (t < -255.0){
          t = -255.0;
        }
	
        if(t > 0) { // Positive values are red
          intensity[3*(i*w+j)+0] = 255; // RED component
          intensity[3*(i*w+j)+1] = 255-t; // GREEN component
          intensity[3*(i*w+j)+2] = 255-t; // BLUE component
        } else { // Negative values are blue
          intensity[3*(i*w+j)+0] = 255+t; // RED component
          intensity[3*(i*w+j)+1] = 255+t; // GREEN component
          intensity[3*(i*w+j)+2] = 255; // BLUE component
        }
	
      }
    }
    
    
    // Draw a green border on right and bottom, overwrites data being plotted
    for(int i=0;i<h;++i){
      intensity[3*(i*w+w-1)+0] = 0;     // RED component
      intensity[3*(i*w+w-1)+1] = 255;   // GREEN component
      intensity[3*(i*w+w-1)+2] = 0;     // BLUE component
    }
    for(int i=0;i<w;++i){
      intensity[3*((h-1)*w+i)+0] = 0;   // RED component
      intensity[3*((h-1)*w+i)+1] = 255; // GREEN component
      intensity[3*((h-1)*w+i)+2] = 0;   // BLUE component
    }

    liveVizDeposit(m, sx,sy, w,h, intensity, this);
    delete[] intensity;
  }



};

#include "wave2d.def.h"
