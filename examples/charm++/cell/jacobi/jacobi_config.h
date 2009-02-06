#ifndef __JACOBI_CONFIG_H__
#define __JACOBI_CONFIG_H__


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
//   DDD : Local matrix data
//   MMM : Holds the maxError for a sub-matrix per iteration (filled in by doCalculation())
//   NNN : Is the nothern ghost data
//   SSS : Is the southern ghost data
//   WWW : Is the western ghost data
//   EEE : Is the eastern ghost data
//   FFF : Is a flag set for the first chare array element (most north-west element).  For all
//           other elements, it is not set. (Used by doCalculation() to know which elements
//           should be held constant and thus not updated each iteration).
//   --- : Unused (forced to 0)
//


////////////////////////////////////////////////////////////////////////////////
// Configuration Defines set by User

#define NUM_ROWS      58  // The number of data rows each chare has
#define NUM_COLS      62  // The number of data columns each chare has (vectorized code used on SPE if this is a '(multiple of 4) +/- 2')
#define NUM_CHARES    32  // The number of chares (per dimension)

#define MAX_ERROR  0.001f  // The value that all errors have to be below for the program to finish

#define DISPLAY_MATRIX          0
#define DISPLAY_MAX_ERROR_FREQ  5

#define USE_REDUCTION                   (1)
#define USE_MESSAGES                    (0)
#define CHARE_MAPPING_TO_PES__STRIPE    (1)
#define REPORT_MAX_ERROR_BUFFER_DEPTH  (16)


////////////////////////////////////////////////////////////////////////////////
// Generated Configuration Defines and Utility Defines

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


////////////////////////////////////////////////////////////////////////////////
// Configuration Checks 

#if (NUM_ROWS <= 0)
  #error "NUM_ROWS must be greater than 0"
#endif

#if (NUM_COLS <= 0)
  #error "NUM_COLS must be greater than 0"
#endif

#if (((NUM_COLS % 2) != 0) || ((NUM_COLS % 4) == 0))
  #error "NUM_COLS must be a multiple of 2 but not a multiple of 4 (for vectorized code)"
#endif

// DMK - TODO : FIXME - Architecture specific check... Currently the Offload API has a
//   maximum limit on the size of each buffer that is passed to a work request.  If this is a
//   Cell architecture, check to make sure the tile size is not too large to DMA.  Once the
//   Offload API is able to handle large DMAs, this check can be removed.
// NOTE: Constant of 4 comes from sizeof(float) since floats are used for the calculation.
#if ((CMK_CELL != 0) && ((DATA_BUFFER_SIZE * 4) > SPE_DMA_LIST_ENTRY_MAX_LENGTH))
  #error "Matrix size per chare is too large (cannot DMA to SPE).  Reduce NUM_COLS and/or NUM_ROWS."
#endif

// DMK - TODO : FIXME - Architecture specific check... Currently, heterogeneous architectures cannot
//   properly auto-generate pack/unpack routines for some messages.  If this is a heterogeneous
//   architecture, do not use messages (force parameter marshalling).
#if ((CMK_HETERO_SUPPORT != 0) && (USE_MESSAGES != 0))
  #undef USE_MESSAGES
  #define USE_MESSAGES (0)
  #warning "USE_MESSAGES cannot be enabled for hetergeneous architectures... disabling USE_MESSAGES..."
#endif


#endif //__JACOBI_CONFIG_H__
