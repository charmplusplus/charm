/*-------------------------------------------------------------
 *  file   : lvServer.C
 *  author : Isaac Dooley
 *  date   : 08-18-05
 *  
 *  A sample LiveViz polling server application. Uses the new
 *  poll mode interface.
 *  
 *  Use the corresponding lvClient java application. The server
 *  in poll mode will not expect a ccs "lvConfig" request, so 
 *  it will not work with other standard liveViz clients which 
 *  are not in poll mode. This may change at some point in the 
 *  future.
 *  
 *-------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include "liveViz.h"
#include "lvServer.decl.h"
#include "math.h"

int nChares=4;
int gridWidth=400;  // image dimensions
int gridHeight=400; // image dimensions

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_lvServer arr;

/*mainchare*/
class Main : public CBase_Main
{
public:

  Main(CkArgMsg* m)
  {
	delete m;
	
	CkPrintf("lvServer with %d slices on %d Processors\n", nChares, CkNumPes());
	mainProxy=thishandle;
	
	// Register with python
	mainProxy.registerPython("pycode");
	
	// Register with liveViz
	liveVizPollInit();

    // Create Array
	CkArrayOptions opts(nChares);
	arr = CProxy_lvServer::ckNew(opts);
	CkPrintf("array has been created\n");

	// At this point all chares are just idle until a liveViz message comes
  } 


  // can be called by client generated python
  void pycall(int handle){
	arr.genNextImage();
 	pythonReturn(handle);
  }
  
};


/*array [1D]*/
class lvServer : public CBase_lvServer 
{
public:
  lvServer(CkMigrateMessage *m) {}
  lvServer() {}


  void genNextImage() {
	int myHeight = gridHeight / nChares;  // dimensions of my portion of the image
    int myWidth = gridWidth;

	int sx=0, sy=thisIndex*(myHeight);    // start coordinate(upper left corner) for my portion of the image

    unsigned char *intensity= new unsigned char[3*myWidth*myHeight]; // the byte array that is my portion of the image

	// Fill in the image byte array
    for (int y=0;y<myHeight;y++)
      for (int x=0;x<myWidth;x++) {
    	int i=y*myWidth+x;
		if(thisIndex==0){
		  intensity[3*i+0] = (byte) 128;//Red
		  intensity[3*i+1] = (byte) 128;//Green
		  intensity[3*i+2] = (byte) 128;//Blue
		}
		else if(thisIndex==1){
		  intensity[3*i+0] = (byte) 0;//Red
		  intensity[3*i+1] = (byte) 0;//Green
		  intensity[3*i+2] = (byte) 0;//Blue
		}
		else{
		  intensity[3*i+0] = (byte) 128;//Red
		  intensity[3*i+1] = (byte) (128*thisIndex/nChares);//Green
		  intensity[3*i+2] = (byte) (128*thisIndex/nChares);//Blue
		}

    }

	// Deposit with liveViz
	liveVizPollDeposit((ArrayElement *)this, 
					   sx,sy, 
					   myWidth,myHeight, 
					   gridWidth, gridHeight, 
					   (byte*) intensity,
					   sum_image_data, 3);

	// cleanup
    delete[] intensity;
  }
  
};
  
#include "lvServer.def.h"
