#include <stdio.h>

#include "jacobi.decl.h"
#include "jacobi.h"
#include "jacobi_shared.h"

#include "main.decl.h"


extern /* readonly */ CProxy_Main mainProxy;


Jacobi::Jacobi() {

  // Init member variables
  ghostCount = 0;
  ghostCountNeeded = 4;
  int chareX = GET_CHARE_X(thisIndex);
  int chareY = GET_CHARE_Y(thisIndex);
  if (chareX == 0) ghostCountNeeded--;
  if (chareY == 0) ghostCountNeeded--;
  if (chareX == NUM_CHARES - 1) ghostCountNeeded--;
  if (chareY == NUM_CHARES - 1) ghostCountNeeded--;

  // Allocate memory for the buffers
  // NOTE: Each buffer will have enough room for all the data (local data + ghost data from bordering chares)
  matrix = (volatile float*)malloc_aligned(sizeof(float) * DATA_BUFFER_SIZE, 16);
  matrixTmp = (volatile float*)malloc_aligned(sizeof(float) * DATA_BUFFER_SIZE, 16);

  // Initialize the data
  memset((float*)matrix, 0, sizeof(float) * DATA_BUFFER_SIZE);
  memset((float*)matrixTmp, 0, sizeof(float) * DATA_BUFFER_SIZE);

  // If this is the first element, set it's matrix[DATA_OFFSET] to 1.0f (this is the only fixed point)
  if (thisIndex == 0) {
    matrixTmp[DATA_OFFSET] = matrix[DATA_OFFSET] = 1.0f;
    matrixTmp[DATA_BUFFER_COLS - 1] = matrix[DATA_BUFFER_COLS - 1] = 1.0f;  // Flag the first element's matrices
  }

  // Init the iteration counter to zero
  iterCount = 0;
}

Jacobi::Jacobi(CkMigrateMessage *msg) {
}

Jacobi::~Jacobi() {
  if (matrix != NULL) { free_aligned((void*)matrix); matrix = NULL; }
  if (matrixTmp != NULL) { free_aligned((void*)matrixTmp); matrixTmp = NULL; }
}

void Jacobi::startIteration() {

  // Send ghost data to the neighbors
  int chareX = GET_CHARE_X(thisIndex);
  int chareY = GET_CHARE_Y(thisIndex);

  // Send to the north
  if (chareY > 0) {
    thisProxy[GET_CHARE_I(chareX, chareY-1)].southData(NUM_COLS, (float*)matrix + DATA_NORTH_DATA_OFFSET, iterCount);
  }

  // Send to the south
  if (chareY < (NUM_CHARES - 1)) {
    thisProxy[GET_CHARE_I(chareX, chareY+1)].northData(NUM_COLS, (float*)matrix + DATA_SOUTH_DATA_OFFSET, iterCount);
  }

  // Send to the west
  if (chareX > 0) {
    float buf[NUM_ROWS];
    for (int i = 0; i < NUM_ROWS; i++)
      buf[i] = matrix[DATA_BUFFER_COLS * i + DATA_WEST_DATA_OFFSET];
    thisProxy[GET_CHARE_I(chareX - 1, chareY)].eastData(NUM_ROWS, buf, iterCount);
  }

  // Send to the east
  if (chareX < (NUM_CHARES - 1)) {
    float buf[NUM_ROWS];
    for (int i = 0; i < NUM_ROWS; i++)
      buf[i] = matrix[DATA_BUFFER_COLS * i + DATA_EAST_DATA_OFFSET];
    thisProxy[GET_CHARE_I(chareX + 1, chareY)].westData(NUM_ROWS, buf, iterCount);
  }
}

void Jacobi::northData(int size, float* ghostData, int iterRef) {
  // Check to see if this message has arrived in order...
  if (iterCount == iterRef) {  // If so, process it
    memcpy((float*)matrix + DATA_NORTH_BUFFER_OFFSET, ghostData, NUM_COLS * sizeof(float));
    attemptCalculation();
  } else {                     // If not, resend to self and try again later
    thisProxy[thisIndex].northData(size, ghostData, iterRef);
  }
}

void Jacobi::southData(int size, float* ghostData, int iterRef) {
  // Check to see if this message has arrived in order...
  if (iterCount == iterRef) {  // If so, process it
    memcpy((float*)matrix + DATA_SOUTH_BUFFER_OFFSET, ghostData, NUM_COLS * sizeof(float));
    attemptCalculation();
  } else {                     // If not, resend to self and try again later
    thisProxy[thisIndex].southData(size, ghostData, iterRef);
  }
}

void Jacobi::eastData(int size, float* ghostData, int iterRef) {
  // Check to see if this message has arrived in order...
  if (iterCount == iterRef) {  // If so, process it
    for (int i = 0; i < NUM_ROWS; i++)
      matrix[DATA_BUFFER_COLS * i + DATA_EAST_BUFFER_OFFSET] = ghostData[i];
    attemptCalculation();
  } else {                     // If not, resend to self and try again later
    thisProxy[thisIndex].eastData(size, ghostData, iterRef);
  }
}

void Jacobi::westData(int size, float* ghostData, int iterRef) {
  // Check to see if this message has arrived in order...
  if (iterCount == iterRef) {  // If so, process it
    for (int i = 0; i < NUM_ROWS; i++)
      matrix[DATA_BUFFER_COLS * i + DATA_WEST_BUFFER_OFFSET] = ghostData[i];
    attemptCalculation();
  } else {                     // If not, resend to self and try again later
    thisProxy[thisIndex].westData(size, ghostData, iterRef);
  }
}

void Jacobi::attemptCalculation() {

  ghostCount++;
  if (ghostCount >= ghostCountNeeded) {
    // Reset ghostCount for the next iteration
    // NOTE: No two iterations can overlap because of the reduction
    ghostCount = 0;

    // Send a message so the threaded doCalculation() entry method will be called
    thisProxy[thisIndex].doCalculation();  // NOTE: Message needed because doCalculation is [threaded].
                                           //   DO NOT call doCalculation directly!
  }
}


#if USE_CALLBACK != 0
  void doCalculation_callback(void* obj) { ((Jacobi*)obj)->doCalculation_post(); }
#endif


void Jacobi::doCalculation() {

  // Send the work request to the Offload API
  #if USE_CALLBACK == 0

    WRHandle wrHandle = sendWorkRequest(FUNC_DoCalculation,
                                        NULL, 0,                                              // readWrite data
                                        (float*)matrix, sizeof(float) * DATA_BUFFER_SIZE,     // readOnly data
                                        (float*)matrixTmp, sizeof(float) * DATA_BUFFER_SIZE,  // writeOnly data
                                        CthSelf()
                                       );

    disableTrace();

    if (wrHandle == INVALID_WRHandle)
      CkPrintf("Jacobi[%d]::doCalculation() - ERROR - sendWorkRequest() returned INVALID_WRHandle\n", thisIndex);
    else
      CthSuspend();

    doCalculation_post();

  #else

    WRHandle wrHandle = sendWorkRequest(FUNC_DoCalculation,
                                        NULL, 0,                                      // readWrite data
                                        matrix, sizeof(float) * DATA_BUFFER_SIZE,     // readOnly data
                                        matrixTmp, sizeof(float) * DATA_BUFFER_SIZE,  // writeOnly data
                                        this,
                                        WORK_REQUEST_FLAGS_NONE,
                                        doCalculation_callback
                                       );
    if (wrHandle == INVALID_WRHandle)
      CkPrintf("Jacobi[%d]::doCalculation() - ERROR - sendWorkRequest() returned INVALID_WRHandle\n", thisIndex);

  #endif
}

void Jacobi::doCalculation_post() {

  // Get the maxError calculated by the work request and contribute it the reduction for this overall iteration
  contribute(sizeof(float), (float*)matrixTmp, CkReduction::max_float);


  // Display the matrix
  #if DISPLAY_MATRIX != 0
    printf("matrix[%d] @ %p = {\n", thisIndex, matrix);
    for (int y = 0; y < DATA_BUFFER_ROWS; y++) {
      printf("  ");
      for (int x = 0; x < DATA_BUFFER_COLS; x++) {
        printf(" %f", matrix[GET_DATA_I(x,y)]);
      }
      printf("\n");
    }
    printf("}\n");
  #endif

  #if DISPLAY_MATRIX != 0
    printf("matrixTmp[%d] @ %p = {\n", thisIndex, matrixTmp);
    for (int y = 0; y < DATA_BUFFER_ROWS; y++) {
      printf("  ");
      for (int x = 0; x < DATA_BUFFER_COLS; x++) {
        printf(" %f", matrixTmp[GET_DATA_I(x,y)]);
      }
      printf("\n");
    }
    printf("}\n");
  #endif


  // Swap the matrix and matrixTmp pointers
  volatile float *tmp = matrix;
  matrix = matrixTmp;
  matrixTmp = tmp;

  // Start the next iteration for this chare
  iterCount++;
  startIteration();
}


#include "jacobi.def.h"
