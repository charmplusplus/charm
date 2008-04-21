#include "main.decl.h"

#define DEBUG 1

#define DEFAULT_PARTICLES 1000
#define DEFAULT_N 3
#define DEFAULT_M 3
#define DEFAULT_RADIUS 10
#define DEFAULT_FINALSTEPCOUNT 1

/* readonly */ CProxy_Main mainProxy;
/* readonly */ CProxy_Cell cellArray; /* CHECKME */
/* readonly */ CProxy_Interaction interactionArray; /* CHECKME */

/* readonly */ int particles;
/* readonly */ int m;
/* readonly */ int n;
/* readonly */ double radius;
/* readonly */ int finalStepCount; 

/* FIXME */
/*
struct Particle {
  double x,y,z; // coordinates
};*/

class Cell : public CBase_Cell {
  private:
    //VECTOR OF PARTICLES /* FIXME */
    int forceCount; // to count the returns from interactions
    int stepCount;  // to count the number of steps, and decide when to stop

  public:
    Cell();
    Cell(CkMigrateMessage *msg);
    ~Cell();

    void start();
    void force();
    void stepDone();
};

class Interaction : public CBase_Interaction {
  private:
    // VARIABLES FOR FOCES COMPUTATION /* FIXME */
    int cellCount;  // to count the number of interact() calls
    
    /* FIXME */
    int bufferedX;
    int bufferedY;

  public:
    Interaction();
    Interaction(CkMigrateMessage *msg);

    void interact(/*CkVec<CkArrayIndex1D> particles,*/ int i, int j);

};
    

class Main : public CBase_Main {

  private:
    int checkinCount;
    
    int doneCount; // Count to terminate

    #ifdef DEBUG
      int interactionCount;
    #endif

  public:

    /// Constructors ///
    Main(CkArgMsg* msg);
    Main(CkMigrateMessage* msg);

    void done();
};

// Entry point of Charm++ application
Main::Main(CkArgMsg* msg) {
  
  particles = DEFAULT_PARTICLES;
  m = DEFAULT_M;
  n = DEFAULT_N;
  radius = DEFAULT_RADIUS;
  finalStepCount = DEFAULT_FINALSTEPCOUNT;

  delete msg;
  doneCount = 0;

  #ifdef DEBUG
    interactionCount=0;
  #endif

  mainProxy = thisProxy;

  cellArray = CProxy_Cell::ckNew(m,n);

  interactionArray = CProxy_Interaction::ckNew();

  int i, j, k, l;
  for (int x = 0; x < m ; x++ ) {
    for (int y = 0; y < n; y++ ) {

      //Processor Round Robin needed
 
      #ifdef DEBUG
        CkPrintf("INITIAL:( %d, %d) ( %d , %d )\n", x,y,x,y);
        interactionCount++;
      #endif

      // self interaction
      interactionArray( x, y, x, y ).insert( /* processor number */0 );

      // (x,y) and (x+1,y) pair
      (x == m-1) ? (i=(x+1)%m, k=x) : (i=x, k=x+1);
      #ifdef DEBUG
        CkPrintf("INITIAL:( %d, %d) ( %d , %d )\n", i,y,k,y);
        interactionCount++;
      #endif
      interactionArray( i, y, k, y ).insert( /* processor number */0 );

      // (x,y) and (x,y+1) pair
      (y == n-1) ? (j=(y+1)%n, l=y) : (j=y, l=y+1);
      #ifdef DEBUG
        CkPrintf("INITIAL:( %d, %d) ( %d , %d )\n", x,j,x,l);
        interactionCount++;
      #endif
      interactionArray( x, j, x, l ).insert( /* processor number */0 );

      // (x,y) and (x+1,y+1) pair, Irrespective of y /* UNDERSTAND */
      (x == m-1) ? ( i=(x+1)%m, k=x, j=(y+1)%n, l=y ) : (i=x, k=x+1, j=y, l=(y+1)%n );
      #ifdef DEBUG
        CkPrintf("INITIAL:( %d, %d) ( %d , %d )\n", i,j,k,l);
        interactionCount++;
      #endif
      interactionArray( i, j, k, l ).insert( /* processor number */0 );

      // (x,y) and (x-1,y+1) pair /* UNDERSTAND */
      (x == 0) ? ( i=x, k=(x-1+m)%m, j=y, l=(y+1)%n ) : (i=x-1, k=x, j=(y+1)%n, l=y );
      #ifdef DEBUG
        CkPrintf("INITIAL:( %d, %d) ( %d , %d )\n", i,j,k,l);
        interactionCount++;
      #endif
      interactionArray( i, j, k, l ).insert( /* processor number */0 );

    }
  }

  interactionArray.doneInserting();
  #ifdef DEBUG
    CkPrintf("Interaction Count: %d\n", interactionCount);
  #endif

  cellArray.start();
}

// Constructor needed for chare object migration (ignore for now)
// NOTE: This constructor does not need to appear in the ".ci" file
Main::Main(CkMigrateMessage* msg) { }

void Main::done() {
  doneCount ++;
  if( doneCount >= m*n) {
    CkExit();
  }
}


Cell::Cell() {
  forceCount = 0;
  stepCount = 0;
}

// Constructor needed for chare object migration (ignore for now)
// NOTE: This constructor does not need to appear in the ".ci" file
Cell::Cell(CkMigrateMessage *msg) { }                                         
Cell::~Cell() {
  /* FIXME */
  // Deallocate Atom lists
}


void Cell::start() {

  int x = thisIndex.x;
  int y = thisIndex.y;

  int i, j, k, l;

  #ifdef DEBUG
    CkPrintf("START:( %d, %d) ( %d , %d )\n", x,y,x,y);
  #endif
  
  // self interaction
  interactionArray( x, y, x, y).interact( x, y);

  // interaction with (x-1, y-1)
  (x == 0) ? ( i=x, k=(x-1+m)%m, j=y, l=(y-1+n)%n ) : (i=x-1, k=x, j=(y-1+n)%n, l=y);
  interactionArray( i, j, k, l ).interact( x, y);

  // interaction with (x-1, y)
  (x == 0) ? (i=x, k=(x-1+m)%m) : (i=x-1, k=x);
  interactionArray( i, y, k, y).interact( x, y);

  // interaction with (x-1, y+1)
  (x == 0) ? ( i=x, k=(x-1+m)%m, j=y, l=(y+1)%n ) : (i=x-1, k=x, j=(y+1)%n, l=y);
  interactionArray( i, j, k, l ).interact( x, y);


  // interaction with (x, y-1)
  (y == 0) ? (j=y, l=(y-1+n)%n) : (j=(y-1+n)%n, l=y);
  interactionArray( x, j, x, l ).interact( x, y);

  // interaction with (x, y+1)
  (y == n-1) ? (j=(y+1)%n, l=y) : (j=y, l=y+1);
  interactionArray( x, j, x, l ).interact( x, y);


  // interaction with (x+1, y-1)
  (x == m-1) ? ( i=(x+1)%m, k=x, j=(y-1+n)%n, l=y ) : (i=x, k=x+1, j=y, l=(y-1+n)%n );
  interactionArray( i, j, k, l ).interact( x, y);

  // interaction with (x+1, y)
  (x == m-1) ? (i=(x+1)%m, k=x) : (i=x, k=x+1);
  interactionArray( i, y, k, y).interact( x, y);

  // interaction with (x+1, y+1)
  (x == m-1) ? ( i=(x+1)%m, k=x, j=(y+1)%n, l=y ) : (i=x, k=x+1, j=y, l=(y+1)%n );
  interactionArray( i, j, k, l ).interact( x, y);

}

void Cell::force() {
  forceCount++;
  if( forceCount >= 9) {
    // Received all it's forces from the interactions.
    stepCount++;
    
    /* FIX ME*/
    // Methods to migrate atoms.

    #ifdef DEBUG
    if( forceCount > 9 )
      CkPrintf("ERROR\n");
    #endif

    if(stepCount >= finalStepCount) {
      #ifdef DEBUG
        CkPrintf("STEP: %d DONE:( %d , %d )\n", stepCount, thisIndex.x, thisIndex.y);
      #endif
      mainProxy.done();
    } else {
      thisProxy( thisIndex.x, thisIndex.y ).start();
    }
  }
    
}

Interaction::Interaction() {
  cellCount = 0;

  /* FIXME */
    bufferedX = 0;
    bufferedY = 0;
}

Interaction::Interaction(CkMigrateMessage *msg) { }
  

void Interaction::interact( int x, int y ) {

  if(cellCount == 0) {
    bufferedX = x;
    bufferedY = y;

    // self interaction check
    if( thisIndex.x == thisIndex.z && thisIndex.y == thisIndex.w ) {
      CkPrintf("SELF: ( %d , %d )\n", thisIndex.x, thisIndex.y );
      cellArray( x, y).force();
    }
  }

  cellCount++;

  if( cellCount >= 2) {
    CkPrintf("PAIR:( %d , %d )  ( %d , %d ) \n", bufferedX, bufferedY, x, y );
    cellArray( bufferedX, bufferedY).force();
    cellArray( x, y).force();
  }

}

#include "main.def.h"
