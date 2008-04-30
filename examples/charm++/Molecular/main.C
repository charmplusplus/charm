/*
 University of Illinois at Urbana-Champaign
 Department of Computer Science
 Parallel Programming Lab
 2008
*/

#include "liveViz.h"
#include "common.h"
#include "main.decl.h"
#include "main.h"
#include "cell.decl.h"

/* readonly */ CProxy_Main mainProxy;
/* readonly */ CProxy_Cell cellArray;
/* readonly */ CProxy_Interaction interactionArray;

/* readonly */ int numParts;
/* readonly */ int m; // Number of Chare Rows
/* readonly */ int n; // Number of Chare Columns
/* readonly */ int L; 
/* readonly */ double radius;
/* readonly */ int finalStepCount; 

// Entry point of Charm++ application
Main::Main(CkArgMsg* msg) {
  int i, j, k, l;  

  numParts = DEFAULT_PARTICLES;
  m = DEFAULT_M;
  n = DEFAULT_N;
  L = DEFAULT_L;
  radius = DEFAULT_RADIUS;
  finalStepCount = DEFAULT_FINALSTEPCOUNT;

  delete msg;
  checkInCount = 0;

  mainProxy = thisProxy;

  // initializing the cell 2D array
  cellArray = CProxy_Cell::ckNew(m,n);

  // initializing the interaction 4D array
  interactionArray = CProxy_Interaction::ckNew();
 
  // For Round Robin insertion
  int numPes = CkNumPes();
  int currPE = -1;

  for (int x = 0; x < m ; x++ ) {
    for (int y = 0; y < n; y++ ) {

      // self interaction
      interactionArray( x, y, x, y ).insert( (currPE++) % numPes );

      // (x,y) and (x+1,y) pair
      (x == m-1) ? (i=0, k=x) : (i=x, k=x+1);
      interactionArray( i, y, k, y ).insert( (currPE++) % numPes );

      // (x,y) and (x,y+1) pair
      (y == n-1) ? (j=0, l=y) : (j=y, l=y+1);
      interactionArray( x, j, x, l ).insert( (currPE++) % numPes );

      // (x,y) and (x+1,y+1) pair, Irrespective of y
      (x == m-1) ? ( i=0, k=x, j=(y+1)%n, l=y ) : (i=x, k=x+1, j=y, l=(y+1)%n );
      interactionArray( i, j, k, l ).insert( (currPE++) % numPes );

      // (x,y) and (x-1,y+1) pair
      (x == 0) ? ( i=x, k=(x-1+m)%m, j=y, l=(y+1)%n ) : (i=x-1, k=x, j=(y+1)%n, l=y );
      interactionArray( i, j, k, l ).insert( (currPE++) % numPes );

    }
  }

  interactionArray.doneInserting();

  // setup liveviz
  CkCallback c(CkIndex_Cell::requestNextFrame(0),cellArray);
  liveVizConfig cfg(liveVizConfig::pix_color,true);
  liveVizInit(cfg,cellArray,c);

  sleep(1);
  cellArray.start();
}

// Constructor for chare object migration
Main::Main(CkMigrateMessage* msg) { }

void Main::checkIn() {

  checkInCount ++;
  if( checkInCount >= m*n)
    CkExit();

}


#include "main.def.h"
