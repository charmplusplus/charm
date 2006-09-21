#include <stdio.h>

#include "spert.h"
#include "jacobi_shared.h"
#include "sim_printf.h"


#ifdef __cplusplus
extern "C"
#endif
void funcLookup(int funcIndex,
                void* readWritePtr, int readWriteLen,
                void* readOnlyPtr, int readOnlyLen,
                void* writeOnlyPtr, int writeOnlyLen,
		DMAListEntry* dmaList
               ) {

  switch (funcIndex) {

    case SPE_FUNC_INDEX_INIT: break;   // Not needed, but not invalid (i.e. - don't let the default case get it)
    case SPE_FUNC_INDEX_CLOSE: break;  // Not needed, but not invalid (i.e. - don't let the default case get it)

    case FUNC_DoCalculation: doCalculation((float*)writeOnlyPtr, (float*)readOnlyPtr); break;

    default:
      printf("SPE_%d :: !!!!! WARNING !!!!! :: SPE Received Invalid funcIndex (%d)... Ignoring...\n", (int)getSPEID(), funcIndex);
      break;
  }
}


void doCalculation(volatile float* matrixTmp, volatile float* matrix) {

  float maxError = 0.0f;
  int x, y;

  register int isNorthwestern = (matrix[DATA_BUFFER_COLS - 1] == 1.0f);

  // Peel off the first iteration of the y loop since the starting x values changes depending on
  //   the value of isNorthwestern
  for (x = ((__builtin_expect(isNorthwestern, 0)) ? (2) : (1)); x < (DATA_BUFFER_COLS - 1); x++) {
    register int index = GET_DATA_I(x,1);

    matrixTmp[index] = (matrix[index] +
                        matrix[GET_DATA_I(x - 1, 1)] +
                        matrix[GET_DATA_I(x + 1, 1)] +
                        matrix[GET_DATA_I(x    , 0)] +
                        matrix[GET_DATA_I(x    , 2)]
                       ) / 5.0f;

    register float tmp = matrixTmp[index] - matrix[index];
    // NOTE: Values should rise so matrixTmp[index] should be > matrix[index]
    if (__builtin_expect(tmp < 0.0f, 0)) tmp *= -1.0f;
    if (__builtin_expect(maxError < tmp, 0)) maxError = tmp;
  }

  // Calculate the rest of the values (same regardless of isNorthwestern)
  for (y = 2; y < (DATA_BUFFER_ROWS - 1); y++) {
    for (x = 1; x < (DATA_BUFFER_COLS - 1); x++) {
      register int index = GET_DATA_I(x,y);

      matrixTmp[index] = (matrix[index] +
                          matrix[GET_DATA_I(x - 1, y    )] +
                          matrix[GET_DATA_I(x + 1, y    )] +
                          matrix[GET_DATA_I(x    , y - 1)] +
                          matrix[GET_DATA_I(x    , y + 1)]
		         ) / 5.0f;

      register float tmp = matrixTmp[index] - matrix[index];
      // NOTE: Values should rise so matrixTmp[index] should be > matrix[index]
      if (__builtin_expect(tmp < 0.0f, 0)) tmp *= -1.0f;
      if (__builtin_expect(maxError < tmp, 0)) maxError = tmp;
    }
  }

  // NOTE : Since the writeOnlyPtr is being used, the data buffer is initially allocated
  //   on the SPE and is not initialized.  Make sure to write to all the entries so there
  //   are no "junk" values anywhere in matrixTmp when it is passed back to main memory.
  //   Though, ignore the areas that will be overwritten when ghost data arrives next
  //   iteration.  This is a tradeoff... have to do this or use the readWritePtr.  These
  //   stores to the LS are probably faster than DMAing the matrixTmp buffer down to the LS.  ;)
  matrixTmp[GET_DATA_I(0, DATA_BUFFER_ROWS - 1)] = 0.0f;                     // unused corner (here for completeness)
  matrixTmp[GET_DATA_I(DATA_BUFFER_COLS - 1, DATA_BUFFER_ROWS - 1)] = 0.0f;  // unused corner (here for completeness)
  if (__builtin_expect(isNorthwestern, 0)) {
    matrixTmp[GET_DATA_I(1, 1)] = 1.0f;  // Hold this single element constant across iterations
    matrixTmp[GET_DATA_I(DATA_BUFFER_COLS - 1, 0)] = 1.0f;  // northwest flag
  } else {
    matrixTmp[GET_DATA_I(DATA_BUFFER_COLS - 1, 0)] = 0.0f;  // northwest flag
  }

  // Place the maxError in index 0 of matrixTmp
  matrixTmp[GET_DATA_I(0, 0)] = maxError;
}
