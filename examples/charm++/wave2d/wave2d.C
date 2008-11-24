#include "liveViz.h"
#include "wave2d.decl.h"

// This program solves the 2-d wave equation over a grid, displaying pretty results through liveViz
// The discretization used below is described in the accompanying paper.pdf
// Author: Isaac Dooley 2008

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_Wave arrayProxy;

#define TotalDataWidth  800
#define TotalDataHeight 700
#define chareArrayWidth  4
#define chareArrayHeight  3
#define total_iterations 5000
#define numInitialPertubations 5

#define mod(a,b)  (((a)+b)%b)

enum {
  right,
  left,
  up,
  down,
};


class Main : public CBase_Main
{
public:
  int iteration;
  int count;

  Main(CkArgMsg* m) {
    iteration = 0;
    count = 0;
    mainProxy = thisProxy; // store the main proxy
    
    CkPrintf("Running wave2d on %d processors\n", CkNumPes());

    // Create new array of worker chares
    arrayProxy = CProxy_Wave::ckNew(chareArrayWidth, chareArrayHeight);

    // setup liveviz
    CkCallback c(CkIndex_Wave::requestNextFrame(0),arrayProxy);
    liveVizConfig cfg(liveVizConfig::pix_color,true);
    liveVizInit(cfg,arrayProxy,c);

    //Start the computation
    arrayProxy.begin_iteration();
  }

  // Each worker calls this method
  void iterationCompleted() {
    count++;
    if(count == chareArrayWidth*chareArrayHeight){
      if (iteration == total_iterations) {
	CkPrintf("Program Done!\n");
	CkExit();
      } else { 
	// Start the next iteration
	count = 0;
	iteration++;
	if(iteration % 20 == 0) CkPrintf("Completed %d iterations\n", iteration);    
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
    srand(0); // ensure that the same random numbers are used for each chare array element

    for(int i=0;i<myheight*mywidth;i++)
      pressure[i] = pressure_old[i] = 0.0;
    
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
	    double rscaled = (distanceToCenter/radius)*3.0*3.14159/2.0; // ranges from 0 to 3pi/2						
	    double t = 700.0 * cos(rscaled) ; // Range won't exceed -700 to 700
	    pressure[i*mywidth+j] = pressure_old[i*mywidth+j] = t;
	  }
	}						
      }
    }
  }

  Wave(CkMigrateMessage* m) { } // Migration is not supported in this example

  ~Wave() { }

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
    thisProxy(mod(thisIndex.x-1, chareArrayWidth), thisIndex.y).recvGhosts(right, myheight, left_edge);
    // Send my right edge
    thisProxy(mod(thisIndex.x+1, chareArrayWidth), thisIndex.y).recvGhosts(left, myheight, right_edge);
    // Send my top edge
    thisProxy(thisIndex.x, mod(thisIndex.y-1, chareArrayHeight)).recvGhosts(down, mywidth, top_edge);
    // Send my bottom edge
    thisProxy(thisIndex.x, mod(thisIndex.y+1, chareArrayHeight)).recvGhosts(up, mywidth, bottom_edge);

    delete [] right_edge;
    delete [] left_edge;
    delete [] top_edge;
    delete [] bottom_edge;
  }

  void recvGhosts(int whichSide, int size, double ghost_values[]) {
    
    if(whichSide == right)
      for(int i=0;i<size;++i)
	buffer_right[i] = ghost_values[i];

    else if(whichSide == left)
      for(int i=0;i<size;++i)
	buffer_left[i] = ghost_values[i];
    
    else if(whichSide == down)
      for(int i=0;i<size;++i)
	buffer_down[i] = ghost_values[i];
    
    else if(whichSide == up)    
      for(int i=0;i<size;++i)
	buffer_up[i] = ghost_values[i];
        
    check_and_compute();
  }

  void check_and_compute() {
    if (--messages_due == 0) {

      // Compute the new values based on the current and previous step values

      double *pressure_new = new double[mywidth*myheight];

      for(int i=0;i<myheight;++i){
	for(int j=0;j<mywidth;++j){

	  // Current time's pressures for neighboring array locations
	  double left  = (j==0          ? buffer_left[i]  : pressure[i*mywidth+j-1] );
	  double right = (j==mywidth-1  ? buffer_right[i] : pressure[i*mywidth+j+1] );
	  double up    = (i==0          ? buffer_up[j]    : pressure[(i-1)*mywidth+j] );
	  double down  = (i==myheight-1 ? buffer_down[j]  : pressure[(i+1)*mywidth+j] );

	  // Current time's pressure for this array location
	  double curr = pressure[i*mywidth+j];

	  // Previous time's pressure for this array location
	  double old  = pressure_old[i*mywidth+j];

	  // Compute the future time's pressure for this array location
	  pressure_new[i*mywidth+j] = 0.4*0.4*(left+right+up+down - 4.0*curr)-old+2.0*curr;

	}
      }
		
      // Advance to next step by copying values to the arrays for the previous steps
      for(int i=0;i<myheight;++i)
	for(int j=0;j<mywidth;++j){
	  pressure_old[i*mywidth+j] = pressure[i*mywidth+j];
	  pressure[i*mywidth+j] = pressure_new[i*mywidth+j];
	}
      
      delete[] pressure_new;
      messages_due = 4;
      mainProxy.iterationCompleted();
    }
  }


  // provide my portion of the image to the graphical liveViz client                           
  void requestNextFrame(liveVizRequestMsg *m){

    // Draw my part of the image, plus a nice 1px border along my right/bottom boundary
    int sx=thisIndex.x*mywidth; // where my portion of the image is located
    int sy=thisIndex.y*myheight;
    int w=mywidth; // Size of my rectangular portion of the image
    int h=myheight;
    
    // set the output pixel values for my rectangle
    // Each RGB component is a char which can have 256 possible values.
    unsigned char *intensity= new unsigned char[3*w*h];
    for(int i=0;i<myheight;++i){
      for(int j=0;j<mywidth;++j){

        double p = pressure[i*mywidth+j];
        if(p > 255.0) p = 255.0;    // Keep values in valid range
        if(p < -255.0) p = -255.0;  // Keep values in valid range
        	
        if(p > 0) { // Positive values are red
          intensity[3*(i*w+j)+0] = 255; // RED component
          intensity[3*(i*w+j)+1] = 255-p; // GREEN component
          intensity[3*(i*w+j)+2] = 255-p; // BLUE component
        } else { // Negative values are blue
          intensity[3*(i*w+j)+0] = 255+p; // RED component
          intensity[3*(i*w+j)+1] = 255+p; // GREEN component
          intensity[3*(i*w+j)+2] = 255; // BLUE component
        }
	
      }
    }
    
    // Draw a green border on right and bottom of this chare array's pixel buffer. 
    // This will overwrite some pressure values at these pixels.
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
