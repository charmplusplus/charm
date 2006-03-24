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
  matrix = (float*)malloc_aligned(sizeof(float) * DATA_BUFFER_SIZE, 16);
  matrixTmp = (float*)malloc_aligned(sizeof(float) * DATA_BUFFER_SIZE, 16);

  // Initialize the data
  memset(matrix, 0, sizeof(float) * DATA_BUFFER_SIZE);
  memset(matrixTmp, 0, sizeof(float) * DATA_BUFFER_SIZE);

  // If this is the first element, set it's matrix[DATA_OFFSET] to 1.0f (this is the only fixed point)
  if (thisIndex == 0) {
    matrixTmp[DATA_OFFSET] = matrix[DATA_OFFSET] = 1.0f;
    matrixTmp[DATA_BUFFER_COLS - 1] = matrix[DATA_BUFFER_COLS - 1] = 1.0f;  // Flag the first element's matrices
  }
}

Jacobi::Jacobi(CkMigrateMessage *msg) {
}

Jacobi::~Jacobi() {
  if (matrix != NULL) { free_aligned(matrix); matrix = NULL; }
  if (matrixTmp != NULL) { free_aligned(matrixTmp); matrixTmp = NULL; }
}

void Jacobi::startIteration() {

  // Send ghost data to the neighbors
  int chareX = GET_CHARE_X(thisIndex);
  int chareY = GET_CHARE_Y(thisIndex);

  // DEBUG
  static int debugFlag = 0;
  if (thisIndex == 0 && debugFlag == 0) {
    CkPrintf("Jacobi[0]::startIteration() - Called...\n");
    debugFlag = 1;
  }

  // Send to the north
  if (chareY > 0) {
    thisProxy[GET_CHARE_I(chareX, chareY-1)].southData(NUM_COLS, matrix + DATA_NORTH_DATA_OFFSET);
  }

  // Send to the south
  if (chareY < (NUM_CHARES - 1)) {
    thisProxy[GET_CHARE_I(chareX, chareY+1)].northData(NUM_COLS, matrix + DATA_SOUTH_DATA_OFFSET);
  }

  // Send to the west
  if (chareX > 0) {
    float buf[NUM_ROWS];
    for (int i = 0; i < NUM_ROWS; i++)
      buf[i] = matrix[DATA_BUFFER_COLS * i + DATA_WEST_DATA_OFFSET];
    thisProxy[GET_CHARE_I(chareX - 1, chareY)].eastData(NUM_ROWS, buf);
  }

  // Send to the east
  if (chareX < (NUM_CHARES - 1)) {
    float buf[NUM_ROWS];
    for (int i = 0; i < NUM_ROWS; i++)
      buf[i] = matrix[DATA_BUFFER_COLS * i + DATA_EAST_DATA_OFFSET];
    thisProxy[GET_CHARE_I(chareX + 1, chareY)].westData(NUM_ROWS, buf);
  }
}

void Jacobi::northData(int size, float* ghostData) {
  memcpy(matrix + DATA_NORTH_BUFFER_OFFSET, ghostData, NUM_COLS * sizeof(float));
  attemptCalculation();
}

void Jacobi::southData(int size, float* ghostData) {
  memcpy(matrix + DATA_SOUTH_BUFFER_OFFSET, ghostData, NUM_COLS * sizeof(float));
  attemptCalculation();
}

void Jacobi::eastData(int size, float* ghostData) {
  for (int i = 0; i < NUM_ROWS; i++)
    matrix[DATA_BUFFER_COLS * i + DATA_EAST_BUFFER_OFFSET] = ghostData[i];
  attemptCalculation();
}

void Jacobi::westData(int size, float* ghostData) {
  for (int i = 0; i < NUM_ROWS; i++)
    matrix[DATA_BUFFER_COLS * i + DATA_WEST_BUFFER_OFFSET] = ghostData[i];
  attemptCalculation();
}

void Jacobi::attemptCalculation() {

  ghostCount++;
  if (ghostCount >= ghostCountNeeded) {
    // Reset ghostCount for the next iteration
    // NOTE: No two iterations can overlap because of the reduction
    ghostCount = 0;

    // Send a message so the threaded doCalculation() entry method will be called
    thisProxy[thisIndex].doCalculation();
  }
}

void Jacobi::doCalculation() {

  // Send the work request to the Offload API
  WRHandle wrHandle = sendWorkRequest(FUNC_DoCalculation,
                                      NULL, 0,                                      // readWrite data
                                      matrix, sizeof(float) * DATA_BUFFER_SIZE,     // readOnly data
                                      matrixTmp, sizeof(float) * DATA_BUFFER_SIZE,  // writeOnly data
                                      CthSelf()
                                     );
  if (wrHandle == INVALID_WRHandle)
    CkPrintf("Jacobi[%d]::doCalculation() - ERROR - sendWorkRequest() returned INVALID_WRHandle\n", thisIndex);
  else
    CthSuspend();

  // Get the maxError calculated by the work request and contribute it the reduction for this overall iteration
  contribute(sizeof(float), matrixTmp, CkReduction::max_float);


  // Display the matrix
  #if DISPLAY_MATRIX != 0
    printf("matrix[%d] = {\n", thisIndex);
    for (int y = 0; y < DATA_BUFFER_ROWS; y++) {
      printf("  ");
      for (int x = 0; x < DATA_BUFFER_COLS; x++) {
        printf(" %f", matrix[GET_DATA_I(x,y)]);
      }
      printf("\n");
    }
    printf("}\n");
  #endif

  // Swap the matrix and matrixTmp pointers
  float *tmp = matrix;
  matrix = matrixTmp;
  matrixTmp = tmp;
}


#include "jacobi.def.h"
