#include <stdio.h>

#include "jacobi.decl.h"
#include "jacobi.h"
#include "jacobi_config.h"

#include "main.decl.h"


extern /* readonly */ CProxy_Main mainProxy;
extern /* readonly */ CProxy_Jacobi jacobiProxy;


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
  matrix = (float*)(CmiMallocAligned(sizeof(float) * DATA_BUFFER_SIZE, 128));
  matrixTmp = (float*)(CmiMallocAligned(sizeof(float) * DATA_BUFFER_SIZE, 128));

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

  // Initialize the saved message pointers
  #if USE_MESSAGES != 0
    eastMsgSave[0] = new EastWestGhost();
    eastMsgSave[1] = NULL;
    westMsgSave[0] = new EastWestGhost();
    westMsgSave[1] = NULL;
    northMsgSave[0] = new NorthSouthGhost();
    northMsgSave[1] = NULL;
    southMsgSave[0] = new NorthSouthGhost();
    southMsgSave[1] = NULL;

    futureEastMsg = NULL;
    futureWestMsg = NULL;
    futureNorthMsg = NULL;
    futureSouthMsg = NULL;
  #endif

  // Check in with the main chare
  mainProxy.createdCheckIn();
}

Jacobi::Jacobi(CkMigrateMessage *msg) {
}

Jacobi::~Jacobi() {

  // Clean up the matrix data
  if (matrix != NULL) { CmiFreeAligned((void*)matrix); matrix = NULL; }
  if (matrixTmp != NULL) { CmiFreeAligned((void*)matrixTmp); matrixTmp = NULL; }

  // Clean up any saved messages
  #if USE_MESSAGES != 0
  delete eastMsgSave[0];
  delete eastMsgSave[1];
  delete westMsgSave[0];
  delete westMsgSave[1];
  delete northMsgSave[0];
  delete northMsgSave[1];
  delete southMsgSave[0];
  delete southMsgSave[1];
  #endif
}

void Jacobi::startIteration() {

  // Send ghost data to the neighbors
  int chareX = GET_CHARE_X(thisIndex);
  int chareY = GET_CHARE_Y(thisIndex);

  // Send to the north
  if (chareY > 0) {
    #if USE_MESSAGES != 0
      memcpy(northMsgSave[0]->data, (float*)matrix + DATA_NORTH_DATA_OFFSET, sizeof(float) * NUM_COLS);
      northMsgSave[0]->iterCount = iterCount;
      thisProxy[GET_CHARE_I(chareX, chareY-1)].southData_msg(northMsgSave[0]);
      northMsgSave[0] = NULL;
    #else
      thisProxy[GET_CHARE_I(chareX, chareY - 1)].southData(NUM_COLS, (float*)matrix + DATA_NORTH_DATA_OFFSET, iterCount);
    #endif
  }

  // Send to the south
  if (chareY < (NUM_CHARES - 1)) {
    #if USE_MESSAGES != 0
      memcpy(southMsgSave[0]->data, (float*)matrix + DATA_SOUTH_DATA_OFFSET, sizeof(float) * NUM_COLS);
      southMsgSave[0]->iterCount = iterCount;
      thisProxy[GET_CHARE_I(chareX, chareY+1)].northData_msg(southMsgSave[0]);
      southMsgSave[0] = NULL;
    #else
      thisProxy[GET_CHARE_I(chareX, chareY + 1)].northData(NUM_COLS, (float*)matrix + DATA_SOUTH_DATA_OFFSET, iterCount);
    #endif
  }

  // Send to the west
  if (chareX > 0) {
    #if USE_MESSAGES != 0
      float* dataPtr = westMsgSave[0]->data;
      for (int i = 0; i < NUM_ROWS; i++)
        dataPtr[i] = matrix[DATA_BUFFER_COLS * i + DATA_WEST_DATA_OFFSET];
      westMsgSave[0]->iterCount = iterCount;
      thisProxy[GET_CHARE_I(chareX - 1, chareY)].eastData_msg(westMsgSave[0]);
      westMsgSave[0] = NULL;
    #else
      float buf[NUM_ROWS];
      for (int i = 0; i < NUM_ROWS; i++)
        buf[i] = matrix[DATA_BUFFER_COLS * i + DATA_WEST_DATA_OFFSET];
      thisProxy[GET_CHARE_I(chareX - 1, chareY)].eastData(NUM_ROWS, buf, iterCount);
    #endif
  }

  // Send to the east
  if (chareX < (NUM_CHARES - 1)) {
    #if USE_MESSAGES != 0
      float* dataPtr = eastMsgSave[0]->data;
      for (int i = 0; i < NUM_ROWS; i++)
        dataPtr[i] = matrix[DATA_BUFFER_COLS * i + DATA_EAST_DATA_OFFSET];
      eastMsgSave[0]->iterCount = iterCount;
      thisProxy[GET_CHARE_I(chareX + 1, chareY)].westData_msg(eastMsgSave[0]);
      eastMsgSave[0] = NULL;
    #else
      float buf[NUM_ROWS];
      for (int i = 0; i < NUM_ROWS; i++)
        buf[i] = matrix[DATA_BUFFER_COLS * i + DATA_EAST_DATA_OFFSET];
      thisProxy[GET_CHARE_I(chareX + 1, chareY)].westData(NUM_ROWS, buf, iterCount);
    #endif
  }

  // Process any future messages that have already been received
  // NOTE: Important... this code assumes that startIteration() is called directly from
  //   doCalculation_post()... i.e. iterCount is incremented and then these future
  //   messages are processed before control is passed back to the Charm++ scheduler that
  //   way no messages can be received inbetween (and thus a future message overwritten).
  #if USE_MESSAGES != 0
    if (futureNorthMsg != NULL) { northData_msg(futureNorthMsg); futureNorthMsg = NULL; }
    if (futureSouthMsg != NULL) { southData_msg(futureSouthMsg); futureSouthMsg = NULL; }
    if (futureEastMsg != NULL) { eastData_msg(futureEastMsg); futureEastMsg = NULL; }
    if (futureWestMsg != NULL) { westData_msg(futureWestMsg); futureWestMsg = NULL; }
  #endif
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

void Jacobi::northData_msg(NorthSouthGhost *msg) {
  // Check to see if this message has arrived in order...
  if (msg->iterCount == iterCount) {  // If so, process it
    memcpy((float*)matrix + DATA_NORTH_BUFFER_OFFSET, msg->data, NUM_COLS * sizeof(float));
    northMsgSave[1] = msg;  // Save the message for later use
    attemptCalculation();
  } else if (msg->iterCount == iterCount + 1) {  // For next iteration so save the message
    futureNorthMsg = msg;    
  } else {                            // If not, resend to self and try again later
    thisProxy[thisIndex].northData_msg(msg);
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

void Jacobi::southData_msg(NorthSouthGhost *msg) {
  // Check to see if this message has arrived in order...
  if (msg->iterCount == iterCount) {  // If so, process it
    memcpy((float*)matrix + DATA_SOUTH_BUFFER_OFFSET, msg->data, NUM_COLS * sizeof(float));
    southMsgSave[1] = msg;  // Save the message for later use
    attemptCalculation();
  } else if (msg->iterCount == iterCount + 1) {  // For next iteration so save the message
    futureSouthMsg = msg;    
  } else {                            // If not, resend to self and try again later
    thisProxy[thisIndex].southData_msg(msg);
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

void Jacobi::eastData_msg(EastWestGhost* msg) {
  // Check to see if this message has arrived in order...
  if (msg->iterCount == iterCount) {  // If so, process it
    float* data = msg->data;
    for (int i = 0; i < NUM_ROWS; i++)
      matrix[DATA_BUFFER_COLS * i + DATA_EAST_BUFFER_OFFSET] = data[i];
    eastMsgSave[1] = msg;  // Save the message for later use
    attemptCalculation();
  } else if (msg->iterCount == iterCount + 1) {  // For next iteration so save the message
    futureEastMsg = msg;    
  } else {                            // If not, resend to self and try again later
    thisProxy[thisIndex].eastData_msg(msg);
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

void Jacobi::westData_msg(EastWestGhost* msg) {
  // Check to see if this message has arrived in order...
  if (msg->iterCount == iterCount) {  // If so, process it
    float* data = msg->data;
    for (int i = 0; i < NUM_ROWS; i++)
      matrix[DATA_BUFFER_COLS * i + DATA_WEST_BUFFER_OFFSET] = data[i];
    westMsgSave[1] = msg;  // Save the message for later use
    attemptCalculation();
  } else if (msg->iterCount == iterCount + 1) {  // For next iteration so save the message
    futureWestMsg = msg;    
  } else {                            // If not, resend to self and try again later
    thisProxy[thisIndex].westData_msg(msg);
  }
}

void Jacobi::attemptCalculation() {
  ghostCount++;
  if (ghostCount >= ghostCountNeeded) {
    ghostCount = 0;
    thisProxy[thisIndex].doCalculation();  // NOTE: Message needed because doCalculation is [accel].
  }
}

void Jacobi::doCalculation_post() {

  #if USE_REDUCTION != 0
    // Get the maxError calculated by the work request and contribute it the reduction for this overall iteration
    contribute(sizeof(float), (float*)matrixTmp, CkReduction::max_float);
  #else
    mainProxy.reportMaxError(*((float*)matrixTmp), iterCount); 
  #endif

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
  float *tmp = matrix;
  matrix = matrixTmp;
  matrixTmp = tmp;

  // Start the next iteration for this chare
  iterCount++;

  // Swap the message pointers
  #if USE_MESSAGES != 0
    SWAP(eastMsgSave[0], eastMsgSave[1], EastWestGhost*);
    SWAP(westMsgSave[0], westMsgSave[1], EastWestGhost*);
    SWAP(northMsgSave[0], northMsgSave[1], NorthSouthGhost*);
    SWAP(southMsgSave[0], southMsgSave[1], NorthSouthGhost*);
  #endif

  startIteration();
}


#include "jacobi.def.h"
