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

enum { left=0, right, up, down };

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
    CkArrayOptions opts(chareArrayWidth, chareArrayHeight);
    arrayProxy = CProxy_Wave::ckNew(opts);

    // setup liveviz
    CkCallback c(CkIndex_Wave::requestNextFrame(0),arrayProxy);
    liveVizConfig cfg(liveVizConfig::pix_color,true);
    liveVizInit(cfg,arrayProxy,c, opts);

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

  double *pressure_old;  // time t-1
  double *pressure; // time t
  double *pressure_new;  // time t+1

  double *buffers[4];

  // Constructor, initialize values
  Wave() {

    mywidth=TotalDataWidth / chareArrayWidth;
    myheight= TotalDataHeight / chareArrayHeight;

    pressure_new  = new double[mywidth*myheight];
    pressure = new double[mywidth*myheight];
    pressure_old  = new double[mywidth*myheight];

    buffers[left] = new double[myheight];
    buffers[right]= new double[myheight];
    buffers[up]   = new double[mywidth];
    buffers[down] = new double[mywidth];

    messages_due = 4;

    InitialConditions();
  }


  // Setup some Initial pressure pertubations for timesteps t-1 and t
  void InitialConditions(){
    srand(0); // Force the same random numbers to be used for each chare array element

    for(int i=0;i<myheight*mywidth;i++)
      pressure[i] = pressure_old[i] = 0.0;
    
    for(int s=0; s<numInitialPertubations; s++){    
      // Determine where to place a circle within the interior of the 2-d domain
      int radius = 20+rand() % 30;
      int xcenter = radius + rand() % (TotalDataWidth - 2*radius);
      int ycenter = radius + rand() % (TotalDataHeight - 2*radius);
      // Draw the circle
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

  Wave(CkMigrateMessage* m) { }

  ~Wave() { }

  void begin_iteration(void) {

    double *top_edge = &pressure[0];
    double *bottom_edge = &pressure[(myheight-1)*mywidth];

    double *left_edge = new double[myheight];
    double *right_edge = new double[myheight];
    for(int i=0;i<myheight;++i){
      left_edge[i] = pressure[i*mywidth];
      right_edge[i] = pressure[i*mywidth + mywidth-1];
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
  }
  
  void recvGhosts(int whichSide, int size, double ghost_values[]) {
    for(int i=0;i<size;++i)
      buffers[whichSide][i] = ghost_values[i];   
    check_and_compute();
  }

  void check_and_compute() {
    if (--messages_due == 0) {

      // Compute the new values based on the current and previous step values

      for(int i=0;i<myheight;++i){
	for(int j=0;j<mywidth;++j){

	  // Current time's pressures for neighboring array locations
	  double L = (j==0          ? buffers[left][i]  : pressure[i*mywidth+j-1] );
	  double R = (j==mywidth-1  ? buffers[right][i] : pressure[i*mywidth+j+1] );
	  double U = (i==0          ? buffers[up][j]    : pressure[(i-1)*mywidth+j] );
	  double D = (i==myheight-1 ? buffers[down][j]  : pressure[(i+1)*mywidth+j] );

	  // Current time's pressure for this array location
	  double curr = pressure[i*mywidth+j];

	  // Previous time's pressure for this array location
	  double old  = pressure_old[i*mywidth+j];

	  // Compute the future time's pressure for this array location
	  pressure_new[i*mywidth+j] = 0.4*0.4*(L+R+U+D - 4.0*curr)-old+2.0*curr;

	}
      }
		
      // Advance to next step by shifting the data back one step in time
      double *tmp = pressure_old;
      pressure_old = pressure;
      pressure = pressure_new;
      pressure_new = tmp;

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
