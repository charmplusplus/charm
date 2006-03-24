#ifndef __JACOBI_SHARED_H__
#define __JACOBI_SHARED_H__


//////////////////////////////////////////////////////////////////////////////////////////////
// Defines

// Explination of Data Structure
// Each of the buffers (matrix and matrixTmp) are decomposed as follows.  A single data
//   structure is used so that it may easily be moved to and from the SPEs.  There are
//   two of the data structures, one is read and one is written during each iteration.
//   After an iteration, they are switched so the one that was read during iteration 'i'
//   is written to during iteration 'i+1'.
//
//        |/__ NUM_COLS  __\|
//        |\               /|
//
//    MMM NNN NNN ... NNN NNN FFF  ____
//    WWW DDD DDD ... DDD DDD EEE   /\
//    ... ... ... ... ... ... ...   NUM_ROWS
//    WWW DDD DDD ... DDD DDD EEE  _\/_
//    --- SSS SSS ... SSS SSS ---
//
// Where:
//   MMM : Holds the maxError for a sub-matrix per iteration (filled in by doCalculation on SPE)
//   NNN : Is the nothern ghost data
//   SSS : Is the southern ghost data
//   WWW : Is the western ghost data
//   EEE : Is the eastern ghost data
//   FFF : Is a flag set for the first element (most north-west) so the single constant value
//           is not changed on the SPE.  For all other elements, it is not set.
//

#define NUM_ROWS      4  // The number of data rows each chare has
#define NUM_COLS      4  // The number of data columns each chare has
#define NUM_CHARES    2  // The number of chares (per dimension)

#define MAX_ERROR  0.01f  // The value that all errors have to be below for the program to finish

#define DISPLAY_MATRIX  1


#define DATA_BUFFER_ROWS   (NUM_ROWS + 2)
#define DATA_BUFFER_COLS   (NUM_COLS + 2)
#define DATA_BUFFER_SIZE   (DATA_BUFFER_ROWS * DATA_BUFFER_COLS)

#define DATA_OFFSET        (DATA_BUFFER_COLS + 1)

#define DATA_NORTH_DATA_OFFSET  (1 + DATA_BUFFER_COLS)
#define DATA_SOUTH_DATA_OFFSET  (DATA_BUFFER_COLS * (DATA_BUFFER_ROWS - 2) + 1)
#define DATA_EAST_DATA_OFFSET   (DATA_BUFFER_COLS * 2 - 2)
#define DATA_WEST_DATA_OFFSET   (DATA_BUFFER_COLS + 1)

#define DATA_NORTH_BUFFER_OFFSET  (1)
#define DATA_SOUTH_BUFFER_OFFSET  (DATA_BUFFER_COLS * (DATA_BUFFER_ROWS - 1) + 1)
#define DATA_EAST_BUFFER_OFFSET   (DATA_BUFFER_COLS * 2 - 1)
#define DATA_WEST_BUFFER_OFFSET   (DATA_BUFFER_COLS)

#define GET_DATA_I(x,y)    ( ((y) * DATA_BUFFER_COLS) + (x) )
#define GET_DATA_X(i)      ( (i) % DATA_BUFFER_COLS )
#define GET_DATA_Y(i)      ( (i) / DATA_BUFFER_COLS )

#define GET_CHARE_I(x,y)   ( ((y) * NUM_CHARES) + (x) )
#define GET_CHARE_X(i)     ( (i) % NUM_CHARES )
#define GET_CHARE_Y(i)     ( (i) / NUM_CHARES )

#define FUNC_DoCalculation   (1)


//////////////////////////////////////////////////////////////////////////////////////////////
// Function Prototypes

extern void funcLookup(int funcIndex, void* readWritePtr, int readWriteLen, void* readOnlyPtr, int readOnlyLen, void* writeOnlyPtr, int writeOnlyPtr);
extern void doCalculation(volatile float* matrixTmp, volatile float* matrix);


#endif //__JACOBI_SHARED_H__
