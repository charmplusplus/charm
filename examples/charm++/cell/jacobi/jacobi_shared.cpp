#include <stdio.h>
#include <spu_intrinsics.h>

#include "spert.h"
#include "jacobi_shared.h"
//#include "sim_printf.h"



#if ((DATA_BUFFER_COLS % 4) == 0) && (FORCE_NO_SPE_OPT == 0)

#warning "[FYI] :: Using vectorized code..."

#define DATA_BUFFER_VECTORS  (DATA_BUFFER_COLS / 4)

vector unsigned int valueMask[DATA_BUFFER_VECTORS];

// Create masks for vectorized code that will zero out east and west ghost elements
void initValueMasks() {
  register vector unsigned int onesMask = { 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF };
  valueMask[0] = spu_insert(0x00, onesMask, 0);
  valueMask[DATA_BUFFER_VECTORS - 1] = spu_insert(0x00, onesMask, 3);
  register int i;
  for (i = 1; i < DATA_BUFFER_VECTORS - 1; i++)
    valueMask[i] = onesMask;
}

void doCalculation(volatile float* matrixTmp, volatile float* matrix) {

  register vector float maxError_vect = { 0.0f, 0.0f, 0.0f, 0.0f };
  register int i;

  // Get northWestern flag
  register unsigned int isNorthwestern = (matrix[DATA_BUFFER_COLS - 1] == 1.0f);

  // Create some constants in some registers
  register vector unsigned char westMask = { 0x0c, 0x0d, 0x0e, 0x0f,  0x10, 0x11, 0x12, 0x13,  0x14, 0x15, 0x16, 0x17,  0x18, 0x19, 0x1a, 0x1b };
  register vector unsigned char eastMask = { 0x14, 0x15, 0x16, 0x17,  0x18, 0x19, 0x1a, 0x1b,  0x1c, 0x1d, 0x1e, 0x1f,  0x00, 0x01, 0x02, 0x03 };
  register vector float tmp_multp2 = { 0.2f, 0.2f, 0.2f, 0.2f };
  register vector float tmp_zeros = { 0.0f, 0.0f, 0.0f, 0.0f };
  register vector float tmp_ones = { 1.0f, 1.0f, 1.0f, 1.0f };

  // Peel the first iteration off the loop below (special case for northWestern)
  { i = DATA_BUFFER_COLS;

    register vector float* srcDataPtr = (vector float*)(matrix + i);

    // Load self values
    register vector float selfData = *srcDataPtr;

    // Load north and south data
    register vector float northData = *(srcDataPtr - (DATA_BUFFER_COLS / 4));
    register vector float southData = *(srcDataPtr + (DATA_BUFFER_COLS / 4));

    // Load one vector to the east and west and mix with self data to create
    //   the required values for east and west data
    register vector float westTmpData = *(srcDataPtr - 1);
    register vector float eastTmpData = *(srcDataPtr + 1);
    register vector float westData = spu_shuffle(westTmpData, selfData, westMask);
    register vector float eastData = spu_shuffle(eastTmpData, selfData, eastMask);

    // Add the values together
    register vector float tmp_sum0 = spu_add(selfData, northData);
    register vector float tmp_sum1 = spu_add(southData, westData);
    register vector float tmp_sum2 = spu_add(tmp_sum0, eastData);
    register vector float tmp_sum = spu_add(tmp_sum1, tmp_sum2);

    // Divide by 5
    register vector float tmp_newSelfData_0 = spu_mul(tmp_sum, tmp_multp2);

    // Select elements based on value mask (i.e. - zero out invalid values created
    //   for the west and east ghost elements as a result of vectorization)
    register int tmp_valueMaskIndex = (i % DATA_BUFFER_COLS) >> 2;
    register vector unsigned int tmp_valueMask = valueMask[tmp_valueMaskIndex];
    register vector float tmp_newSelfData_1 = spu_sel(tmp_zeros, tmp_newSelfData_0, tmp_valueMask);

    // Mask out result for constant elements
    register vector unsigned int tmp_nwmask = { 0x00, 0x00, 0x00, 0x00 };
    if (__builtin_expect(isNorthwestern, 0))
      tmp_nwmask = spu_insert((unsigned int)0xFFFFFFFF, tmp_nwmask, 1);
    register vector float newSelfData = spu_sel(tmp_newSelfData_1, tmp_ones, tmp_nwmask);

    // Store the value in the result matrix
    register vector float* destDataPtr = (vector float*)(matrixTmp + i);
    *destDataPtr = newSelfData;

    // Update maxError from these four values
      // Absolute difference of old and new self values
    register vector float tmp_maxError_0 = spu_sub(selfData, newSelfData);
    register vector float tmp_maxError_1 = spu_sub(newSelfData, selfData);
    register vector unsigned int tmp_cmpMask_0 = spu_cmpgt(tmp_maxError_0, tmp_zeros);
    register vector float tmp_maxError_2 = spu_sel(tmp_maxError_1, tmp_maxError_0, tmp_cmpMask_0);
      // Compare to running maxError values and then update maxError
    register vector unsigned int tmp_cmpMask_1 = spu_cmpgt(maxError_vect, tmp_maxError_2);
    maxError_vect = spu_sel(tmp_maxError_2, maxError_vect, tmp_cmpMask_1);
  }

  // Iterate over the output matrix (matrixTmp), filling in values
  // NOTE: Treat the 2D matrix as a 1D array
  for (i = DATA_BUFFER_COLS + 4; i < DATA_BUFFER_COLS * (DATA_BUFFER_ROWS - 1); i+=4) {

    register vector float* srcDataPtr = (vector float*)(matrix + i);

    // Load self values
    register vector float selfData = *srcDataPtr;

    // Load north and south data
    register vector float northData = *(srcDataPtr - (DATA_BUFFER_COLS / 4));
    register vector float southData = *(srcDataPtr + (DATA_BUFFER_COLS / 4));

    // Load one vector to the east and west and mix with self data to create
    //   the required values for east and west data
    register vector float westTmpData = *(srcDataPtr - 1);
    register vector float eastTmpData = *(srcDataPtr + 1);
    register vector float westData = spu_shuffle(westTmpData, selfData, westMask);
    register vector float eastData = spu_shuffle(eastTmpData, selfData, eastMask);

    // Add the values together
    register vector float tmp_sum0 = spu_add(selfData, northData);
    register vector float tmp_sum1 = spu_add(southData, westData);
    register vector float tmp_sum2 = spu_add(tmp_sum0, eastData);
    register vector float tmp_sum = spu_add(tmp_sum1, tmp_sum2);

    // Divide by 5
    register vector float tmp_newSelfData = spu_mul(tmp_sum, tmp_multp2);

    // Select elements based on value mask (i.e. - zero out invalid values created
    //   for the west and east ghost elements as a result of vectorization)
    register int tmp_valueMaskIndex = (i % DATA_BUFFER_COLS) >> 2;
    register vector unsigned int tmp_valueMask = valueMask[tmp_valueMaskIndex];
    register vector float newSelfData = spu_sel(tmp_zeros, tmp_newSelfData, tmp_valueMask);

    // Store the value in the result matrix
    register vector float* destDataPtr = (vector float*)(matrixTmp + i);
    *destDataPtr = newSelfData;

    // Update maxError from these four values
      // Absolute difference of old and new self values
    register vector float tmp_maxError_0 = spu_sub(selfData, newSelfData);
    register vector float tmp_maxError_1 = spu_sub(newSelfData, selfData);
    register vector unsigned int tmp_cmpMask_0 = spu_cmpgt(tmp_maxError_0, tmp_zeros);
    register vector float tmp_maxError_2 = spu_sel(tmp_maxError_1, tmp_maxError_0, tmp_cmpMask_0);
      // Compare to running maxError values and then update maxError
    register vector unsigned int tmp_cmpMask_1 = spu_cmpgt(maxError_vect, tmp_maxError_2);
    maxError_vect = spu_sel(tmp_maxError_2, maxError_vect, tmp_cmpMask_1);
  }

  // Calculate the final maxError and place it in index 0 of matrixTmp
  register float maxError_0 = spu_extract(maxError_vect, 0);
  register float maxError_1 = spu_extract(maxError_vect, 1);
  register float maxError_2 = spu_extract(maxError_vect, 2);
  register float maxError_3 = spu_extract(maxError_vect, 3);
  register float maxError = maxError_0;
  // TODO : Check to see if the compiler is smart enough to create branchless code for this
  maxError = (maxError > maxError_1) ? (maxError) : (maxError_1);
  maxError = (maxError > maxError_2) ? (maxError) : (maxError_2);
  maxError = (maxError > maxError_3) ? (maxError) : (maxError_3);
  matrixTmp[GET_DATA_I(0, 0)] = maxError;

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
}


#else


#warning "[FYI] :: Using non-vectorized code..."

void initValueMasks() { }

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


#endif



#ifdef __cplusplus
extern "C"
#endif
void funcLookup(int funcIndex,
                void* readWritePtr, int readWriteLen,
                void* readOnlyPtr, int readOnlyLen,
                void* writeOnlyPtr, int writeOnlyLen,
		DMAListEntry* dmaList
               ) {

  register int i;

  switch (funcIndex) {

    case SPE_FUNC_INDEX_INIT:
      initValueMasks();
      break;

    case SPE_FUNC_INDEX_CLOSE: break;  // Not needed, but not invalid (i.e. - don't let the default case get it)

    case FUNC_DoCalculation:
    case FUNC_DoCalculation + 1:  // DEBUG - For Timing (allows different colors for even and odd iterations in projections.. but do the same work)

      // DEBUG
      #if WORK_MULTIPLIER > 1
      for (i = 0; i < WORK_MULTIPLIER; i++)
      #endif

      doCalculation((float*)writeOnlyPtr, (float*)readOnlyPtr);
      break;

    default:
      printf("SPE_%d :: !!!!! WARNING !!!!! :: SPE Received Invalid funcIndex (%d)... Ignoring...\n", (int)getSPEID(), funcIndex);
      break;
  }
}
