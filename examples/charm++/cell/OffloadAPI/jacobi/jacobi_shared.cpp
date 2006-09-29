#include <stdio.h>
#include "spert.h"
#include "jacobi_shared.h"




inline void calcRows(ELEM_TYPE* matrixSrc, ELEM_TYPE* matrixDst, WRInfo* wrInfo) {

  // DEBUG
  register int debugI;

  register int x, y;
  register ELEM_TYPE maxError = (ELEM_TYPE)(0.0);
  register int wrIndex = wrInfo->wrIndex;

  // Shift the matrixSrc pointer
  matrixSrc += BUFFER_COLS;

  // DEBUG
  for (debugI = 0; debugI < 1; debugI++) {

  // Special loop for first row (which skips the first column in the first work request)
  for (x = ((wrIndex == 0) ? (3) : (2)); x < (BUFFER_COLS - 2); x++) {

    register int index = GET_I(x, 0);

    //matrixDst[index] = ((matrixSrc[index] +                 // self
    //                     matrixSrc[index - 1]) +            // left
    //                    (matrixSrc[index + 1] +             // right
    //                     matrixSrc[index + BUFFER_COLS]) +  // below
    //                    matrixSrc[index - BUFFER_COLS]      // above
    //                   ) * (ELEM_TYPE)(0.2);

    register ELEM_TYPE tmp_0 = matrixSrc[index];
    register ELEM_TYPE tmp_1 = matrixSrc[index - 1];
    register ELEM_TYPE tmp_2 = matrixSrc[index + 1];
    register ELEM_TYPE tmp_3 = matrixSrc[index - BUFFER_COLS];
    register ELEM_TYPE tmp_4 = matrixSrc[index + BUFFER_COLS];
    register ELEM_TYPE tmp_5 = (ELEM_TYPE)(0.2);

    register ELEM_TYPE tmp_6 = tmp_0 + tmp_1;
    register ELEM_TYPE tmp_7 = tmp_2 + tmp_3;
    register ELEM_TYPE tmp_8 = tmp_6 + tmp_4;
    register ELEM_TYPE tmp_9 = tmp_7 + tmp_8;

    register ELEM_TYPE tmp_10 = tmp_9 * tmp_5;

    matrixDst[index] = tmp_10;

    //register ELEM_TYPE diff = matrixDst[index] - matrixSrc[index];
    //if (__builtin_expect(diff < (ELEM_TYPE)(0.0), 0))
    //  diff *= ((ELEM_TYPE)(-1.0));
    //if (__builtin_expect(maxError < diff, 0))
    //  maxError = diff;

    register ELEM_TYPE diff = tmp_10 - tmp_0;

    if (__builtin_expect(diff < (ELEM_TYPE)(0.0), 0))
      diff *= ((ELEM_TYPE)(-1.0));
    if (__builtin_expect(maxError < diff, 0))
      maxError = diff;
  }

  // Calculate for the remaining rows
  for (y = 1; y < ROWS_PER_WORKREQUEST; y++) {
    for (x = 2; x < (BUFFER_COLS - 2); x++) {

      register int index = GET_I(x, y);

      //matrixDst[index] = ((matrixSrc[index] +                 // self
      //                     matrixSrc[index - 1]) +            // left
      //                    (matrixSrc[index + 1] +             // right
      //                     matrixSrc[index + BUFFER_COLS]) +  // below
      //                    matrixSrc[index - BUFFER_COLS]      // above
      //                   ) * (ELEM_TYPE)(0.2);
      //register ELEM_TYPE diff = matrixDst[index] - matrixSrc[index];
      //if (__builtin_expect(diff < (ELEM_TYPE)(0.0), 0))
      //  diff *= ((ELEM_TYPE)(-1.0));
      //if (__builtin_expect(maxError < diff, 0))
      //  maxError = diff;

      register ELEM_TYPE tmp_0 = matrixSrc[index];
      register ELEM_TYPE tmp_1 = matrixSrc[index - 1];
      register ELEM_TYPE tmp_2 = matrixSrc[index + 1];
      register ELEM_TYPE tmp_3 = matrixSrc[index - BUFFER_COLS];
      register ELEM_TYPE tmp_4 = matrixSrc[index + BUFFER_COLS];
      register ELEM_TYPE tmp_5 = (ELEM_TYPE)(0.2);

      register ELEM_TYPE tmp_6 = tmp_0 + tmp_1;
      register ELEM_TYPE tmp_7 = tmp_2 + tmp_3;
      register ELEM_TYPE tmp_8 = tmp_6 + tmp_4;
      register ELEM_TYPE tmp_9 = tmp_7 + tmp_8;

      register ELEM_TYPE tmp_10 = tmp_9 * tmp_5;

      matrixDst[index] = tmp_10;

      register ELEM_TYPE diff = tmp_10 - tmp_0;

      if (__builtin_expect(diff < (ELEM_TYPE)(0.0), 0))
        diff *= ((ELEM_TYPE)(-1.0));
      if (__builtin_expect(maxError < diff, 0))
        maxError = diff;

    }
  }

  // DEBUG
  }

  // Place the calculated maxError back into the wrInfo structure
  wrInfo->maxError = maxError;

  // Update the write-only buffer's fixed value if need be
  if (__builtin_expect(wrIndex == 0, 0))
    matrixDst[2] = ((ELEM_TYPE)(1.0));
}


#ifdef __cplusplus
extern "C"
#endif
void funcLookup(int funcIndex,
    void* readWritePtr, int readWriteLen,
    void* readOnlyPtr, int readOnlyLen,
    void* writeOnlyPtr, int writeOnlyLen,
    DMAListEntry* dmaList) {

  switch (funcIndex) {

    case SPE_FUNC_INDEX_INIT:  break;
    case SPE_FUNC_INDEX_CLOSE: break;

    case FUNC_CALCROWS:
      calcRows((ELEM_TYPE*)readOnlyPtr, (ELEM_TYPE*)writeOnlyPtr, (WRInfo*)readWritePtr);
      break;

    default:
      printf("ERROR :: Invalid funcIndex (%d)\n",
             funcIndex);
      break;
  }
}
