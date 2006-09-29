#include <stdio.h>
#include <spu_intrinsics.h>
#include "spert.h"
#include "matmul_shared.h"


inline void calc(ELEM_TYPE* A, ELEM_TYPE* B, ELEM_TYPE* C) {

  //register WRRecord* wrRecord = (WRRecord*)C;
  //register int startRow = wrRecord->startRow;
  //register int startCol = wrRecord->startCol;

  // DEBUG
  //printf("SPE_%d :: startRow = %d, startCol = %d...\n", (int)getSPEID(), startRow, startCol);

  // DEBUG
  //printf("SPE_%d :: A = %p, B = %p, C = %p...\n", (int)getSPEID(), A, B, C);

  register int r,c;

  // Fill in C
  for (r = 0; r < NUM_ROWS_PER_WR; r++) {
    for (c = 0; c < NUM_COLS_PER_WR; c++) {

      // Init the pointers
      register vector ELEM_TYPE* APtr = (vector ELEM_TYPE*)(A + (r * MATRIX_A_COLS));
      register vector ELEM_TYPE* BPtr = (vector ELEM_TYPE*)(B + (c * MATRIX_B_ROWS));
      #if USE_DOUBLE == 0
        register vector ELEM_TYPE sumV = { 0.0f, 0.0f, 0.0f, 0.0f };
      #else
        register vector ELEM_TYPE sumV = { 0.0, 0.0 };
      #endif

      //// DEBUG
      //printf("SPE_%d :: Start C value [%d x %d]... APtr = %p, BPtr = %p\n", (int)getSPEID(), r, c, APtr, BPtr);
      //{
      //  register int i;
      //  printf("SPE_%d :: A's Row = { ", (int)getSPEID());
      //  for (i = 0; i < MATRIX_A_COLS; i++) printf("%f ", (double)*(((float*)(APtr)) + i));
      //  printf("}...\n");
      //  printf("SPE_%d :: B's Column = { ", (int)getSPEID());
      //  for (i = 0; i < MATRIX_B_ROWS; i++) printf("%f ", (double)*(((float*)(BPtr)) + i));
      //  printf("}...\n");
      //}

      register int i;
      for (i = 0; i < MATRIX_A_COLS; i += (16 / sizeof(ELEM_TYPE))) {
        register vector ELEM_TYPE aV = *APtr;
        register vector ELEM_TYPE bV = *BPtr;

        // DEBUG
        //printf("SPE :: aV = { %f, %f, %f, %f }\n", spu_extract(aV, 0), spu_extract(aV, 1), spu_extract(aV, 2), spu_extract(aV, 3));
        //printf("SPE :: bV = { %f, %f, %f, %f }\n", spu_extract(bV, 0), spu_extract(bV, 1), spu_extract(bV, 2), spu_extract(bV, 3));

        APtr += 1;
        BPtr += 1;
        sumV = spu_madd(aV, bV, sumV);

        // DEBUG
        //printf("SPE :: sumV = { %f, %f, %f, %f }\n", spu_extract(sumV, 0), spu_extract(sumV, 1), spu_extract(sumV, 2), spu_extract(sumV, 3));
      }

      // Add the elements of the sumV vector together
      #if USE_DOUBLE == 0
        register ELEM_TYPE sum = 0.0f;
        sum += spu_extract(sumV, 0);
        sum += spu_extract(sumV, 1);
        sum += spu_extract(sumV, 2);
        sum += spu_extract(sumV, 3);
      #else
        register ELEM_TYPE sum = 0.0;
        sum += spu_extract(sumV, 0);
        sum += spu_extract(sumV, 1);
      #endif

      // Store in C
      C[c + (r * NUM_COLS_PER_WR)] = sum;

      // DEBUG
      //printf("SPE_%d :: C value [%d x %d] = %f\n", (int)getSPEID(), r, c, sum);
    }
  }
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

    case FUNC_CALC:
      //calc((ELEM_TYPE*)readWritePtr, (ELEM_TYPE*)readOnlyPtr, (ELEM_TYPE*)writeOnlyPtr);
      break;

    default:
      printf("ERROR :: Invalid funcIndex (%d)\n", funcIndex);
      break;
  }
}
