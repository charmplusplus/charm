#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <spert_ppu.h>
#include <sys/time.h>
#include "matmul_shared.h"


////////////////////////////////////////////////////////////////////////////////////////////////
// Global Data

int callbackFlag = FALSE;

ELEM_TYPE A[MATRIX_A_ROWS * MATRIX_A_COLS] __attribute__((aligned(128))) = { ((ELEM_TYPE)(0.0)) };
ELEM_TYPE B[MATRIX_B_ROWS * MATRIX_B_COLS] __attribute__((aligned(128))) = { ((ELEM_TYPE)(0.0)) };
ELEM_TYPE C[MATRIX_C_ROWS * MATRIX_C_COLS] __attribute__((aligned(128))) = { ((ELEM_TYPE)(0.0)) };
char __wrRecord[NUM_WRS_PER_ITER * SIZEOF_16(WRRecord)];
volatile WRRecord* wrRecord[NUM_WRS_PER_ITER];

////////////////////////////////////////////////////////////////////////////////////////////////
// Function Prototypes

int main(int argc, char* argv[]);
void wrCallback(void* ptr);


////////////////////////////////////////////////////////////////////////////////////////////////
// Function Bodies

int main(int argc, char* argv[]) {

  // Display a header for the user
  printf("----- Offload API : Matrix Multiply -----\n");
  printf("  Number of SPEs Used : %d\n", NUM_SPE_THREADS);
  printf("  Matrices : A[%d x %d], B[%d x %d], C[%d x %d]\n",
         MATRIX_A_ROWS, MATRIX_A_COLS, MATRIX_B_ROWS, MATRIX_B_COLS, MATRIX_C_ROWS, MATRIX_C_COLS
        );
  printf("  Work Requests / Iteration : %d\n", NUM_WRS_PER_ITER);
  printf("  [Rows x Cols] / Work Request : [%d x %d]\n", NUM_ROWS_PER_WR, NUM_COLS_PER_WR);
  register int wrDataSize = ((MATRIX_A_COLS * NUM_ROWS_PER_WR) +
                             (MATRIX_B_ROWS * NUM_COLS_PER_WR) +
                             (NUM_ROWS_PER_WR + NUM_COLS_PER_WR)
                            ) * sizeof(ELEM_TYPE);
  float flopsPerIteration = (float)MATRIX_C_ROWS * (float)MATRIX_C_COLS * (float)MATRIX_A_COLS * 2.0f;
  float flopsPerWR = (float)NUM_ROWS_PER_WR * (float)NUM_COLS_PER_WR * (float)MATRIX_A_COLS * 2.0f;

  printf("  Data / Work Request : %d (0x%08X)\n", wrDataSize, wrDataSize);
  printf("  Flops / Work Request : %.3f KFlops\n", flopsPerWR / 1000.0f);
  printf("  Flops / Iteration : %.3f MFlops\n", flopsPerIteration / 1000000.0f);


  // DEBUG
  //printf("  NUM_WRS_PER_ITER = %d\n", NUM_WRS_PER_ITER);
  //printf("  __wrRecord @ %p\n", __wrRecord);
  //printf("  SIZEOF_16(WRRecord) = %d\n", SIZEOF_16(WRRecord));
  //printf("  wrRecord @ %p\n", wrRecord);


  // Set the random number generator's seed value
  srand(0);

  // Initialize the wrRecord pointers
  register int r1,c1;
  for (r1 = 0; r1 < (MATRIX_C_ROWS / NUM_ROWS_PER_WR); r1++) {
    for (c1 = 0; c1 < (MATRIX_C_COLS / NUM_COLS_PER_WR); c1++) {
      register int index = c1 + (r1 * (MATRIX_C_COLS / NUM_COLS_PER_WR));
      wrRecord[index] = (WRRecord*)(__wrRecord + (index * SIZEOF_16(WRRecord)));
      wrRecord[index]->startRow = r1 * NUM_ROWS_PER_WR;
      wrRecord[index]->startCol = c1 * NUM_COLS_PER_WR;

      //// DEBUG
      //printf("PPE :: wrRecord @ %p = { startRow = %d, startCol = %d, ... } ...\n",
      //       wrRecord[index], wrRecord[index]->startRow, wrRecord[index]->startCol
      //      );
    }
  }

  // Fill in A and B with random values
  register int r0,c0;
  for (r0 = 0; r0 < MATRIX_A_ROWS; r0++)
    for (c0 = 0; c0 < MATRIX_A_COLS; c0++)
      A[c0 + (r0 * MATRIX_A_COLS)] = ((rand() % 100) / (rand() % 100));
  // NOTE: For the sake of filling in random numbers, treat B as if it is in row-major form
  for (r0 = 0; r0 < MATRIX_B_ROWS; r0++)
    for (c0 = 0; c0 < MATRIX_B_COLS; c0++)
      B[c0 + (r0 * MATRIX_B_COLS)] = ((rand() % 100) / (rand() % 100));

  // Initialize the Offload API
  InitOffloadAPI(wrCallback);

  // DEBUG
  //enableTrace();

  // Start timing
  timeval startTime;
  gettimeofday(&startTime, NULL);

  // Entry the main loop
  int iterCount = REPEAT_COUNT;
  while (iterCount > 0) {

    // Send all of the work requests for this iteration
    for (int r = 0; r < (MATRIX_C_ROWS / NUM_ROWS_PER_WR); r++) {
      for (int c = 0; c < (MATRIX_C_COLS / NUM_COLS_PER_WR); c++) {

        // NOTE: A stored in row-major form
        //       B stored in col-major form

        register ELEM_TYPE* ARowPtr = A + ((r * NUM_ROWS_PER_WR) * MATRIX_A_COLS);
        register int ADataSize = NUM_ROWS_PER_WR * MATRIX_A_COLS * sizeof(ELEM_TYPE);
        register ELEM_TYPE* BColPtr = B + ((c * NUM_COLS_PER_WR) * MATRIX_B_ROWS);
        register int BDataSize = NUM_COLS_PER_WR * MATRIX_B_ROWS * sizeof(ELEM_TYPE);
        register volatile WRRecord* recordPtr = wrRecord[c + (r * (MATRIX_C_COLS / NUM_COLS_PER_WR))];
        register int CDataSize = NUM_ROWS_PER_WR * NUM_COLS_PER_WR * sizeof(ELEM_TYPE);

        //// DEBUG
        //printf("PPE :: B = %p, BColPtr = %p...\n", B, BColPtr);
        //register int i, j;
        //printf("PPE :: B data = { ");
        //for (j = 0; j < MATRIX_B_COLS; j++)
        //  for (i = 0; i < MATRIX_B_ROWS; i++)
        //    printf("%f ", *(BColPtr + i + (j * MATRIX_B_ROWS)));
        //printf("}...\n");

        sendWorkRequest(FUNC_CALC,                   // Funcion Index
                        ARowPtr, ADataSize,          // Read/Write Data
                        BColPtr, BDataSize,          // Read-Only Data
                        (void*)recordPtr, CDataSize, // Write-Only Data
                        (void*)recordPtr,            // User Data
                        WORK_REQUEST_FLAGS_RW_IS_RO  // Flags
                       );
      }
    }

    // Wait for them to complete (barrier for all work requests)
    while (callbackFlag == FALSE) OffloadAPIProgress();
    callbackFlag = FALSE;  // reset the flag

    // Decrement the iteration count
    iterCount--;

    // DEBUG
    //displayLastWRTimes();
  }

  // Stop Timing
  timeval endTime;
  gettimeofday(&endTime, NULL);

  // Close the Offload API
  CloseOffloadAPI();

  // Display the matrices
  #if DISPLAY_MATRICES != 0
    register int r2, c2;
    printf("matrix A [%d x %d] = {\n", MATRIX_A_ROWS, MATRIX_A_COLS);
    for (r2 = 0; r2 < MATRIX_A_ROWS; r2++) {
      printf("   ");
      for (c2 = 0; c2 < MATRIX_A_COLS; c2++) {
        register int index = c2 + (r2 * MATRIX_A_COLS);
        #if USE_DOUBLE == 0
          printf("%05.3f ", A[index]);
        #else
          printf("%05.3lf ", A[index]);
        #endif
      }
      printf("\n");
    }
    printf("matrix B [%d x %d] = {\n", MATRIX_B_ROWS, MATRIX_B_COLS);
    for (r2 = 0; r2 < MATRIX_B_ROWS; r2++) {
      printf("   ");
      for (c2 = 0; c2 < MATRIX_B_COLS; c2++) {
        register int index = r2 + (c2 * MATRIX_A_ROWS);
        #if USE_DOUBLE == 0
          printf("%05.3f ", B[index]);
        #else
          printf("%05.3lf ", B[index]);
        #endif
      }
      printf("\n");
    }
    printf("matrix C [%d x %d] = {\n", MATRIX_C_ROWS, MATRIX_C_COLS);
    for (r2 = 0; r2 < MATRIX_C_ROWS; r2++) {
      printf("   ");
      for (c2 = 0; c2 < MATRIX_C_COLS; c2++) {
        register int index = c2 + (r2 * MATRIX_A_COLS);
        #if USE_DOUBLE == 0
          printf("%05.3f ", C[index]);
        #else
          printf("%05.3lf ", C[index]);
        #endif
      }
      printf("\n");
    }
  #endif // DISPLAY_MATRICES != 0

  // Calculate the total flops
  float totalGFlops = (float)REPEAT_COUNT * flopsPerIteration / 1000000000.0f;
  printf("Total Flops : %.3lf GFlops\n", totalGFlops);

  // Calculate the time taken
  double startTimeD = (double)startTime.tv_sec + ((double)startTime.tv_usec / 1000000.0);
  double endTimeD = (double)endTime.tv_sec + ((double)endTime.tv_usec / 1000000.0);
  double timeDiff = endTimeD - startTimeD;
  printf("Time Taken : %.6lf secs\n", timeDiff);
  printf("Average GFlops/s : %.6lf\n", (double)totalGFlops / timeDiff);


  // All Good
  return EXIT_SUCCESS;
}


void wrCallback(void* ptr) {

  static int completeCounter = 0;
  register WRRecord* wrRecord = (WRRecord*)ptr;

  // DEBUG
  //printf("PPE :: wrRecord @ %p = { startRow = %d, startCol = %d, ... } ...\n",
  //       wrRecord, wrRecord->startRow, wrRecord->startCol
  //      );

  // Display text from time to time for the user
  #if DISPLAY_WR_FINISH_FREQ != 0
    if (completeCounter % DISPLAY_WR_FINISH_FREQ == 0)
      printf("PPE :: [INFO] :: completeCounter = %d...\n", completeCounter);
  #endif

  // Merge the results into the C matrix
  for (int r = 0; r < NUM_ROWS_PER_WR; r++)
    for (int c = 0; c < NUM_COLS_PER_WR; c++) {
      register int gIndex = (wrRecord->startCol + c) + ((wrRecord->startRow + r) * MATRIX_C_COLS);
      register int lIndex = c + (r * NUM_COLS_PER_WR);

      // DEBUG
      //printf("PPE ::   [%d x %d] = %f : gIndex = %d, lIndex = %d\n", r, c, wrRecord->C[lIndex], gIndex, lIndex);

      C[gIndex] = wrRecord->C[lIndex];
    }

  // Count this completion
  completeCounter++;

  // Check to see if all work requests for the iteration have completed
  if (completeCounter >= NUM_WRS_PER_ITER) {
    completeCounter = 0;
    callbackFlag = TRUE;
  }  
}
