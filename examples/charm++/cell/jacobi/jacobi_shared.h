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
//        |___ NUM_COLS  ___|
//        |                 |
//
//    MMM NNN NNN ... NNN NNN FFF  ___
//    WWW DDD DDD ... DDD DDD EEE   |
//    ... ... ... ... ... ... ...   NUM_ROWS
//    WWW DDD DDD ... DDD DDD EEE  _|_
//    --- SSS SSS ... SSS SSS ---
//
// Where:
//   DDD : Matrix data
//   MMM : Holds the maxError for a sub-matrix per iteration (filled in by doCalculation on SPE)
//   NNN : Is the nothern ghost data
//   SSS : Is the southern ghost data
//   WWW : Is the western ghost data
//   EEE : Is the eastern ghost data
//   FFF : Is a flag set for the first element (most north-west) so the single constant value
//           is not changed on the SPE.  For all other elements, it is not set.
//   --- : Ignored
//

#define NUM_ROWS      62  // The number of data rows each chare has
#define NUM_COLS      58  // The number of data columns each chare has (vectorized code used on SPE if this is a '(multiple of 4) +/- 2')
#define NUM_CHARES    16  // The number of chares (per dimension)

#define MAX_ERROR  0.001f  // The value that all errors have to be below for the program to finish

#define DISPLAY_MATRIX          0
#define DISPLAY_MAX_ERROR_FREQ  0

#define DATA_BUFFER_ROWS   (NUM_ROWS + 2)
#define DATA_BUFFER_COLS   (NUM_COLS + 2)
#define DATA_BUFFER_EXTRA  (NUM_ROWS * 2)  // Two extra columns worth of data so SPE can collect data for east and west ghosts
#define DATA_BUFFER_WEST_COL_OFFSET  (DATA_BUFFER_ROWS * DATA_BUFFER_COLS)
#define DATA_BUFFER_EAST_COL_OFFSET  (DATA_BUFFER_ROWS * DATA_BUFFER_COLS + NUM_ROWS)
#define DATA_BUFFER_SIZE   (DATA_BUFFER_ROWS * DATA_BUFFER_COLS + DATA_BUFFER_EXTRA)

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

#define REPORT_MAX_ERROR_BUFFER_DEPTH  16

// NOTE: If setting USE_CALLBACK to 0, then the doCalculation() entry methods should
//   be marked as '[threaded]' in the jacobi.ci file.
#define USE_CALLBACK  1

#define USE_REDUCTION  1

#define USE_MESSAGES 1

#define CHARE_MAPPING_TO_PES__STRIPE  1

#define WORK_MULTIPLIER  24

#define FORCE_NO_SPE_OPT  0



#endif //__JACOBI_SHARED_H__
