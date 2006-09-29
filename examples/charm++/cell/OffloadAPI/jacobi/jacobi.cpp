#include <stdio.h>
#include <string.h>
#include <spert_ppu.h>
#include <sys/time.h>
#include "jacobi_shared.h"

////////////////////////////////////////////////////////////////////////////////////////////////
// Global Data

int callbackFlag = FALSE;
ELEM_TYPE maxError;

ELEM_TYPE __matrix0[BUFFER_ROWS * BUFFER_COLS] __attribute__((aligned(128))) = { ((ELEM_TYPE)(0.0)) };
ELEM_TYPE __matrix1[BUFFER_ROWS * BUFFER_COLS] __attribute__((aligned(128))) = { ((ELEM_TYPE)(0.0)) };
char __wrInfo[SIZEOF_16(WRInfo) * NUM_WORKREQUESTS];


////////////////////////////////////////////////////////////////////////////////////////////////
// Function Prototypes

int main(int argc, char* argv[]);
void wrCallback(void* ptr);


////////////////////////////////////////////////////////////////////////////////////////////////
// Function Bodies

int main(int argc, char* argv[]) {

  // Display a header for the user
  printf("----- Offload API : Jacobi -----\n");
  printf("  Number of SPEs Used : %d\n", NUM_SPE_THREADS);
  printf("  Matrix : [%d x %d]\n", MATRIX_ROWS, MATRIX_COLS);
  printf("  Work Requests / Iteration : %d\n", NUM_WORKREQUESTS);
  printf("  Rows / Work Request : %d\n", ROWS_PER_WORKREQUEST);
  register int wrDataSize = (BUFFER_COLS * (ROWS_PER_WORKREQUEST + 1) * sizeof(ELEM_TYPE)) * 2;
  int flopsPerIteration = (MATRIX_ROWS * MATRIX_COLS * 6);
  printf("  Data / Work Request : %d (0x%08X)\n", wrDataSize, wrDataSize);
  printf("  Flops / Iteration : %.3f MFlops\n", (float)flopsPerIteration / 1000000.0f);

  // Initialize the Offload API
  InitOffloadAPI(wrCallback);

  // Create the matrix of values
  volatile ELEM_TYPE* matrixSrc = __matrix0 + BUFFER_COLS;
  volatile ELEM_TYPE* matrixDst = __matrix1 + BUFFER_COLS;

  // Set the fixed value
  matrixSrc[2] = ((ELEM_TYPE)(1.0));

  // Create the array of work request info structures
  volatile WRInfo* wrInfo[NUM_WORKREQUESTS];
  for (int i = 0; i < NUM_WORKREQUESTS; i++) {
    register volatile WRInfo* ptr = (WRInfo*)(__wrInfo + (SIZEOF_16(WRInfo) * i));
    wrInfo[i] = ptr;
    ptr->wrIndex = i;
  }

  // DEBUG
  //enableTrace();

  // Start timing
  timeval startTime;
  gettimeofday(&startTime, NULL);

  // Entry the main loop
  int keepLooping = TRUE;
  int iterCount = 0;
  while (keepLooping == TRUE) {

    // Reset the overall maxError
    maxError = (ELEM_TYPE)(0.0);

    // Send all of the work requests for this iteration
    for (int i = 0; i < NUM_WORKREQUESTS; i++) {
      register int elemOffset = BUFFER_COLS * ROWS_PER_WORKREQUEST * i;
      register ELEM_TYPE* matrixSrcStart = ((ELEM_TYPE*)matrixSrc) + (elemOffset - BUFFER_COLS);
      register ELEM_TYPE* matrixDstStart = ((ELEM_TYPE*)matrixDst) +  elemOffset;
      register int matrixSrcSize = ((BUFFER_COLS * (ROWS_PER_WORKREQUEST + 2)) * sizeof(ELEM_TYPE));
      register int matrixDstSize = ((BUFFER_COLS * ROWS_PER_WORKREQUEST) * sizeof(ELEM_TYPE));

      sendWorkRequest(FUNC_CALCROWS,
                      (void*)wrInfo[i], SIZEOF_16(WRInfo),  // read/write
                      matrixSrcStart, matrixSrcSize, // read-only
                      matrixDstStart, matrixDstSize, // write-only
                      (void*)(wrInfo[i])
                     );
    }

    // Wait for them to complete (barrier for all work requests)
    while (callbackFlag == FALSE) OffloadAPIProgress();
    callbackFlag = FALSE;  // reset the flag

    // Increment the iteration count
    iterCount++;

    // DEBUG
    displayLastWRTimes();

    // Display the maxError for this iteration
    printf("For iteration %d, maxError = %f...\n", iterCount, maxError);

    // Check to see if maxError is small enough to exit
    if (maxError <= MAX_ERROR) {

      // Stop Timing
      timeval endTime;
      gettimeofday(&endTime, NULL);

      printf("Final maxError : %.12f\n", maxError);
      printf("Total Iterations : %d\n", iterCount);
      float totalGFlops = (float)iterCount * (float)flopsPerIteration / 1000000000.0f;
      printf("Total Flops : %.3lf GFlops\n", totalGFlops);

      // Calculate the time taken
      double startTimeD = (double)startTime.tv_sec + ((double)startTime.tv_usec / 1000000.0);
      double endTimeD = (double)endTime.tv_sec + ((double)endTime.tv_usec / 1000000.0);
      double timeDiff = endTimeD - startTimeD;

      printf("Time Taken : %.6lf secs\n", timeDiff);
      printf("Average GFlops/s : %.6lf\n", (double)totalGFlops / timeDiff);

      keepLooping = FALSE;
    }

    // Swap the src and dst arrays
    register volatile ELEM_TYPE* tmpPtr = matrixSrc;
    matrixSrc = matrixDst;
    matrixDst = tmpPtr;
  }


  // Close the Offload API
  CloseOffloadAPI();

  // All Good
  return EXIT_SUCCESS;
}


void wrCallback(void* ptr) {

  static int completeCounter = 0;

  // Count this completion
  completeCounter++;

  // Update the overall maxError
  register int wrIndex = ((volatile WRInfo*)ptr)->wrIndex;
  register ELEM_TYPE localMaxError = ((volatile WRInfo*)ptr)->maxError;

  // DEBUG
  //printf("PPE :: wrCallback() - Called... completeCounter = %d, wrIndex = %d, localMaxError = %f...\n",
  //       completeCounter, wrIndex, localMaxError
  //      );

  if (localMaxError > maxError) maxError = localMaxError;

  // Check to see if all work requests for the iteration have completed
  if (completeCounter >= NUM_WORKREQUESTS) {
    completeCounter = 0;
    callbackFlag = TRUE;
  }  
}
