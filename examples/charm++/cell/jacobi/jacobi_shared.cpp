#include <stdio.h>

#include "jacobi_shared.h"


void funcLookup(int funcIndex,
                void* readWritePtr, int readWriteLen,
                void* readOnlyPtr, int readOnlyLen,
                void* writeOnlyPtr, int writeOnlyLen
               ) {

  switch (funcIndex) {

    case FUNC_DoCalculation: doCalculation((float*)writeOnlyPtr, (float*)readOnlyPtr); break;

    default:
      printf("!!! WARNING !!! :: SPE Received Invalid funcIndex (%d)... Ignoring...\n", funcIndex);
      break;
  }
}


void doCalculation(volatile float* matrixTmp, volatile float* matrix) {

  float maxError = 0.0f;

  // Update matrixTmp with new values
  register int isNorthwestern = (matrix[DATA_BUFFER_COLS - 1] == 1.0f);
  register int startX = ((isNorthwestern) ? (2) : (1));
  for (int y = 1; y < (DATA_BUFFER_ROWS - 1); y++) {
    for (int x = startX; x < (DATA_BUFFER_COLS - 1); x++) {

      matrixTmp[GET_DATA_I(x,y)] = (matrix[GET_DATA_I(x    , y    )] +
                                    matrix[GET_DATA_I(x - 1, y    )] +
                                    matrix[GET_DATA_I(x + 1, y    )] +
                                    matrix[GET_DATA_I(x    , y - 1)] +
                                    matrix[GET_DATA_I(x    , y + 1)]
				   ) / 5.0f;

      float tmp = matrixTmp[GET_DATA_I(x,y)] - matrix[GET_DATA_I(x,y)];
      if (tmp < 0.0f) tmp *= -1.0f;
      if (maxError < tmp) maxError = tmp;
    }
    startX = 1;
  }

  // NOTE : Since the writeOnlyPtr is being used, the data buffer is initially allocated
  //   on the SPE and is not initialized.  Make sure to write to all the entries so there
  //   are no "junk" values anywhere in matrixTmp when it is passed back to main memory.
  //   Though, ignore the areas that will be overwritten when ghost data arrives next
  //   iteration.  This is a tradeoff... have to do this or use the readWritePtr.  These
  //   stores to the LS are probably faster than DMAing the matrixTmp buffer down to the LS.  ;)
  matrixTmp[GET_DATA_I(0, DATA_BUFFER_ROWS - 1)] = 0.0f;                     // unused corner (here for completeness)
  matrixTmp[GET_DATA_I(DATA_BUFFER_COLS - 1, DATA_BUFFER_ROWS - 1)] = 0.0f;  // unused corner (here for completeness)
  if (isNorthwestern) {
    matrixTmp[GET_DATA_I(1, 1)] = 1.0f;  // Hold this single element constant across iterations
    matrixTmp[GET_DATA_I(DATA_BUFFER_COLS - 1, 0)] = 1.0f;  // northwest flag
  } else {
    matrixTmp[GET_DATA_I(DATA_BUFFER_COLS - 1, 0)] = 0.0f;  // northwest flag
  }

  // Place the maxError in index 0 of matrixTmp
  matrixTmp[GET_DATA_I(0, 0)] = maxError;
}
