#include "statcoll.h"
#include "amr.h"
#define CHK_PT_PUP
#define STATS(x) CkPrintf x

/*********************************************************************
 **Method Class: AmrCoordinator
 **Method Name: AmrCooridinator (Default Constructor)
 **Arguments: None
 **Return Type: None
 **Author: Puneet Narula
 **Description: Intialize the parameters and start the creation of the
 **             tree
 *********************************************************************/

AmrCoordinator :: AmrCoordinator(_DMsg* msg)
{
  delete msg;
  arrayProxy = CProxy_Cell2D::ckNew();
  //  CkPrintf("AmrCoordinator on PE %d gets array ID %d\n",
  //	   CkMyPe(),arrayProxy.ckGetGroupID());
  synchInterval = 50;
  depth = 2;
  dimension = 2;
  totalIterations = 500;
  myHandle = thishandle;
  statCollection = 0;
  create_tree();
  phase = 0;
  phaseStep = 0;
}

/*********************************************************************
 **Method Class: AmrCoordinator
 **Method Name: AmrCooridinator 
 **Arguments: StartUpMsg
 **Return Type: None
 **Author: Puneet Narula
 **Description: Intialize the parameters using the values supplied by the
 **user in StartUpMsg and start the creation of the tree
 *********************************************************************/
AmrCoordinator :: AmrCoordinator(StartUpMsg *msg)
{

  synchInterval = msg->synchInterval;
  depth = msg->depth;
  dimension = msg->dimension;
  totalIterations = msg->totalIterations;
  statCollection  = msg->statCollection;
  delete msg;
  switch (dimension){
  case 1:
    arrayProxy =  CProxy_Cell1D::ckNew();
    break;
  case 2:
    arrayProxy =  CProxy_Cell2D::ckNew();
    break;
  case 3:
    arrayProxy =  CProxy_Cell3D::ckNew();
    break;
  }
  myHandle = thishandle;
  /*If stat collection is required*/
  if(statCollection) {
    _CreateStatCollMsg *grpMsg = new _CreateStatCollMsg(myHandle);
    gid =  CProxy_StatCollector::ckNew(grpMsg);
  }
  phase = 0;
  phaseStep = 0;
  startTime = CkWallTimer();
  create_tree();
}

/*********************************************************************
 **Method Class: AmrCoordinator
 **Method Name: synchronise
 **Arguments: _RedMsg
 **Return Type: None
 **Author: Puneet Narula
 **Description: This method is invoked whenever synchronisation takes 
 **place. Different action is taken on the basis of type of the reduction.
 ** RedMsg->type    Action
 **--------------------------
 **      0          Check for Refine in each leaf node by invoking refineExec
 **      1          Refine Completed. Send a message to resume computation
 **      2          End of the iterations. So exit
 *********************************************************************/
void AmrCoordinator :: synchronise(_RedMsg *msg)
{
  if(msg->type == 0){
    //ready for refine
    delete msg;
    if ((phase % 2) != 0) {
      //refine 
      ++phase;
      resetClock();
      STATS(("Refining..\n"));
      //CkPrintf("Coordinator: Recieved refine synch send refineExec\n");
      arrayProxy.refineExec(new _DMsg);
    }
    else {
      //load balance
      ++phase;
      resetClock();
      STATS(("Load Balancing...\n"));
      arrayProxy.goToAtSync(new _DMsg);
    }

  }
  else if (msg->type == 1) { //refine or load balance completed
    delete msg;
    resetClock();
    DEBUGS(("Broadcasting a message to resume\n"));
    STATS(("Doing Iterations...\n"));
    arrayProxy.resume(new _DMsg);
  }
  else if (msg->type == 2) { //end of all iterations
    //    double endTime = CkWallTimer();
    resetClock();
    STATS(("Completed iterations\n"));
    //CkPrintf("Size of bitvector %d\n",sizeof(bitvec));
    // CkPrintf("Time taken %lf\n",endTime-startTime);
   
    delete msg;

    if(statCollection) {
      CProxy_StatCollector grpProxy(gid);
      grpProxy.sendStat(new _DummyMsg);
      leaves = 0;
      migrations = 0;
      refine = 0;
      arefine = 0;
      statMsgs = 0;
    }
    else
      CkExit();

  }
}

void AmrCoordinator :: resetClock()
{
  double endTime = CkWallTimer();
  STATS(("Phase Step  %d Time taken %lf\n",++phaseStep,endTime-startTime));
  startTime = CkWallTimer();
}

/*********************************************************************
 **Method Class: AmrCoordinator
 **Method Name: create_tree
 **Arguments: None
 **Return Type: None
 **Author: Puneet Narula
 **Description: Start the creation of the tree by inserting the root
 *********************************************************************/

void AmrCoordinator :: create_tree()
{
  BitVec root;
  if(depth < 1) 
    CkError("Initial depth should atleast be 1, Current depth %d\n", depth);
  for(int i = 0; i< dimension;i++)
    root.vec[i] = 0;
  root.numbits = 0;
  _ArrInitMsg *msg = new _ArrInitMsg;
  msg->type = 'r';
  msg->parent = root;
  msg->depth = depth;
  msg->interval = synchInterval;
  msg->totalIterations = totalIterations;
  msg->coordHandle = myHandle;
  msg->statCollection = statCollection;
  //msg->grpProxy = grpProxy;
  msg->gid = gid;
  //  DEBUGR(("Inserting %d %d bits %d\n",root.vec[0],root.vec[1],root.numbits));
  CkArrayIndexBitVec index(root);
  switch(dimension) {
  case 1: {
    CProxy_Cell1D aProxy = *(CProxy_Cell1D *) &arrayProxy;
    aProxy[index].insert(msg);
    aProxy.doneInserting();
    break;
  }
  case 2: {
    CProxy_Cell2D aProxy = *(CProxy_Cell2D *) &arrayProxy;
    aProxy[index].insert(msg);
    aProxy.doneInserting();
    break;
  }
  case 3: {
    CProxy_Cell3D aProxy = *(CProxy_Cell3D *) &arrayProxy;
    aProxy[index].insert(msg);
    aProxy.doneInserting();
    break;
  }
  }
  
}

void AmrCoordinator::reportStats(_StatCollMsg* msg)
{
  leaves += msg->msgExpected;
  refine += msg->refineCount;
  arefine += msg->aRefineCount;
  migrations += msg->migCount;

  CkPrintf("PE %d : refines %d autorefines %d migrations %d leaves %d\n",
	   msg->pnum,
	   msg->refineCount,
	   msg->aRefineCount,
	   msg->migCount,
	   msg->msgExpected);
  delete msg;
  if(++statMsgs == CkNumPes()) {
    CkPrintf("Total Leaves %d \n", leaves);
    CkPrintf("Total Refines %d \n", refine);
    CkPrintf("Total Auto Refines %d \n", arefine);
    CkPrintf("Total Migrations %d \n", migrations);
    CkExit();
  }
}

/*
***************************************
*NeighborMsg methods
***************************************
*/


/*********************************************************************
 **Method Class: NeighborMsg
 **Method Name: pack
 **Arguments: Pointer to a NeighborMsg
 **Return Type: pointer to an array (void*)
 **Author: Puneet Narula
 **Description: Serialize the NeighborMsg and return the pointer to the 
 **beginning of serial buffer
 *********************************************************************/

void* NeighborMsg :: pack(NeighborMsg *msg)
{
  int bufSize = msg->dataSize;
  bufSize += (4*sizeof(int));
  bufSize += sizeof(BitVec);
  char *buf = (char *) CkAllocBuffer(msg,bufSize);
  memcpy(buf,&(msg->which_neighbor),sizeof(int));
  buf += sizeof(int);
  memcpy(buf,&(msg->run_until),sizeof(int));
  buf += sizeof(int);
  memcpy(buf,&(msg->numbits),sizeof(int));
  buf += sizeof(int);
  memcpy(buf,&(msg->nborIdx),sizeof(BitVec));
  buf += sizeof(BitVec);
  memcpy(buf,&(msg->dataSize),sizeof(int));
  buf += sizeof(int);
  memcpy(buf,msg->data,msg->dataSize);
  buf -= 4*sizeof(int);
  buf -= sizeof(BitVec);
  delete msg;
  return (void *) buf;
}

/*********************************************************************
 **Method Class: NeighborMsg
 **Method Name: unpack
 **Arguments: Pointer to a serial buffer
 **Return Type: pointer to NeighborMSg 
 **Author: Puneet Narula
 **Description: Read the data from the serial buffer to recreate the
 **Neighbor Msg from it
 *********************************************************************/
NeighborMsg* NeighborMsg :: unpack(void* inbuf)
{

  char *buf = (char *) inbuf;
  NeighborMsg *msg = (NeighborMsg *) CkAllocBuffer(inbuf,sizeof(NeighborMsg));
  
  memcpy(&(msg->which_neighbor),buf,sizeof(int));
  buf += sizeof(int);
  memcpy(&(msg->run_until),buf,sizeof(int));
  buf += sizeof(int);
  memcpy(&(msg->numbits),buf,sizeof(int));
  buf += sizeof(int);
  memcpy(&(msg->nborIdx),buf,sizeof(BitVec));
  buf += sizeof(BitVec);
  memcpy(&(msg->dataSize),buf,sizeof(int));
  buf += sizeof(int);
  msg->data = (void *)new char[msg->dataSize];
  memcpy(msg->data,buf,msg->dataSize);

  CkFreeMsg(inbuf);
  return msg;
}

/*********************************************************************
 **Method Class: NeighborMsg
 **Method Name: pup
 **Arguments: PUP::er
 **Return Type: void
 **Author: Puneet Narula
 **Description: Pack and Unpack the message for migrations and 
 **checkpointing
 *********************************************************************/

void NeighborMsg :: pup(PUP::er &p)
{
  p(which_neighbor);
  p(run_until);
  p(numbits);
  p(dataSize);
  nborIdx.pup(p);
  if(p.isUnpacking())
    data = (void *) new char[dataSize];
  p((char *)data,dataSize);
}

/*
***************************************
*ChildInitMsg methods
***************************************
*/

/*********************************************************************
 **Method Class: ChildInitMsg
 **Method Name: pack
 **Arguments: Pointer to a ChildInitMsg
 **Return Type: pointer to an array (void*)
 **Author: Puneet Narula
 **Description: Serialize the ChildInitMsg and return the pointer to the 
 **beginning of serial buffer
 *********************************************************************/

void* ChildInitMsg :: pack(ChildInitMsg *msg)
{
  int bufSize = msg->dataSize;
  bufSize += 4*sizeof(int);
  char *buf = (char *) CkAllocBuffer(msg,bufSize);
  memcpy(buf,&(msg->num_neighbors),sizeof(int));
  buf += sizeof(int);
  memcpy(buf,&(msg->run_until),sizeof(int));
  buf += sizeof(int);
  memcpy(buf,&(msg->synchstep),sizeof(int));
  buf += sizeof(int);
  memcpy(buf,&(msg->dataSize),sizeof(int));
  buf += sizeof(int);
  memcpy(buf,msg->data,msg->dataSize);
  //  if (!msg->copyToSerialBuffer(buf))
  // CkError("Error: Pack()--Unable to copy the user data NeighborMsg to serial buffer\n");
  buf -= 4*sizeof(int);
  delete msg;
  return (void *) buf;
}


/*********************************************************************
 **Method Class: ChildInitMsg
 **Method Name: unpack
 **Arguments: Pointer to a serial buffer
 **Return Type: pointer to ChildInitMSg 
 **Author: Puneet Narula
 **Description: Read the data from the serial buffer to recreate the
 **ChildInitMsg from it
 *********************************************************************/
ChildInitMsg* ChildInitMsg :: unpack(void* inbuf)
{
  //should get the message already allocated from  the derived class i.e
  //user data to be communicated to the user
  char *buf = (char *) inbuf;
  ChildInitMsg *msg = (ChildInitMsg *) CkAllocBuffer(inbuf,sizeof(ChildInitMsg));
  
  memcpy(&(msg->num_neighbors),buf,sizeof(int));
  buf += sizeof(int);
  memcpy(&(msg->run_until),buf,sizeof(int));
  buf += sizeof(int);
  memcpy(&(msg->synchstep),buf,sizeof(int));
  buf += sizeof(int);
  memcpy(&(msg->dataSize),buf,sizeof(int));
  buf += sizeof(int);
  msg->data = new char[msg->dataSize];
  memcpy(msg->data,buf,msg->dataSize);

  CkFreeMsg(inbuf);
  return msg;
}

/*
*****************************************
*Cell Class Methods
*****************************************
*/

/*********************************************************************
 **Method Class: Cell
 **Method Name: powerOfTwo
 **Arguments: exponent of type integer
 **Return Type: integer
 **Author: Puneet Narula
 **Description: Compute the power of 2 for the exponent given as the 
 **argument and return the result
 *********************************************************************/

int Cell :: powerOfTwo(int exp)
{
  int result = 1;
  //for (int i = 0; i < exp; i++)
  //  result *= 2;
  result = result << exp;
  return result;
}


/*********************************************************************
 **Method Class: Cell
 **Method Name: init_cell
 **Arguments: pointer to _ArrInitMsg
 **Return Type: None
 **Author: Puneet Narula
 **Description: Called by the constructors of the subclasses to initialize
 **the data memebers using the argument
 *********************************************************************/

void Cell :: init_cell(_ArrInitMsg *msg)
{
  int i;

  int temp;
  //for load balancing
  usesAtSync = true;

  //Initailze the fields in the array element of class Cell
  type = msg->type; 
  parent = msg->parent;
  
  synchinterval  = msg->interval;
  synchstep = synchinterval;
  synchleavrep = 0;

  //number of iterations for the program
  run_done = msg->totalIterations;    
  

  myIndex = thisIndex;	//get the bytes for my index
  userData = NULL;

  //DEBUGR(("My index x %d y %d z %d bits %d\n",myIndex.vec[0],
  //  myIndex.vec[1],myIndex.vec[2],myIndex.numbits));

  num_neighbors = 0;
  neighbors_reported = 0;
  
  coordHandle = msg->coordHandle;
  //create an array to hold the count for received messages from
  //neighbors. required because the message may be split if the neighbor
  //is more refined than self
  nborRecvMsgCount = new int [2*dimension];
  for(i=0;i<2*dimension;i++)
    nborRecvMsgCount[i] = 0;

  //create a buffer to buffer the split message temporarily till
  //complete message is received then it is forwarded to the user
  temp = powerOfTwo(dimension -1);
  nborRecvMsgBuff = new NeighborMsg** [2*dimension];
  for(i=0;i<2*dimension;i++) {
    nborRecvMsgBuff[i] = new NeighborMsg* [temp];
    for(int j=0;j<temp;j++)
      nborRecvMsgBuff[i][j] = NULL;
  }
  
  run_until = 0;
  refined = 0;
  autorefine = 0;
  justRefined = 0;

  //create a queue to store the out of order messages
  msg_queue = FIFO_Create();
  start_ptr = NULL;

  //create children array
  children = new BitVec* [dimension];
  for (i = 0; i < dimension; i++) {
    temp = (i==0)? 2: powerOfTwo(i);
    children[i] = new BitVec [temp];
  }

  //neighbors array
  neighbors = new int [2*dimension];
  statCollection = msg->statCollection;
  gid = msg->gid;

  if(statCollection && type == 'l')
  {
    CProxy_StatCollector grpProxy(gid);
    StatCollector* grp = grpProxy.ckLocalBranch();
    grp->registerMe();
  }

}


/*********************************************************************
 **Method Class: Cell
 **Method Name: treeSetup
 **Arguments: pointer to _ArrInitMsg
 **Return Type: None
 **Author: Puneet Narula
 **Description: Called by the constructors of the subclasses to setup
 **the tree initially at the beginning of the program
 *********************************************************************/

void Cell :: treeSetup(_ArrInitMsg *msg)
{
  int size = powerOfTwo(dimension);
  if (dimension == 1)
    size = 2;

  if((type == 'r' || type == 'n') && msg->depth > 0) {
    // root or node with depth > 0
    //This Part is executed only during the initial creation of the tree
    
    //send messages to create nodes further 
    _ArrInitMsg **cmsg;
    cmsg = new _ArrInitMsg* [size];
    for(int i=0; i<size ;i++) {
      cmsg[i] = new _ArrInitMsg;
      cmsg[i]->parent = myIndex;
      cmsg[i]->type = 'n';
      cmsg[i]->interval = synchinterval;
      cmsg[i]->depth  = (msg->depth) - 1;
      cmsg[i]->totalIterations = run_done;
      cmsg[i]->coordHandle  = coordHandle;
      cmsg[i]->statCollection = statCollection;
      cmsg[i]->gid = gid;
    }
    
    create_children(cmsg);
  } 
  else if(type == 'n' && msg->depth == 0) {
    // If I am a node and the depth is zero
    // that is I am actually a leaf so change my type to leaf
    // and create virtual leaves as my children
    // this part is also executed during the intial creation of the tree
    
    //create messages to be sent to my children who are virtual leaves
    _ArrInitMsg **cmsg;
    cmsg = new _ArrInitMsg* [size];
    for(int i=0; i<size ;i++) {
      cmsg[i] = new _ArrInitMsg;
      cmsg[i]->parent = myIndex;
      cmsg[i]->type = 'v';
      cmsg[i]->interval = synchinterval;
      cmsg[i]->depth  = (msg->depth) - 1;
      cmsg[i]->totalIterations = run_done;
      cmsg[i]->coordHandle  = coordHandle;
      cmsg[i]->statCollection = statCollection;
      cmsg[i]->gid = gid;
    }
    
    create_children(cmsg);
    //change my type to leaf
    type = 'l';
  }

}

/*********************************************************************
 **Method Class: Cell
 **Method Name: sendInDimension
 **Arguments: dimension to send(integer), side(+ or -), Pointer to 
 ** Neighbor Msg to be sent to the neigbor
 **Return Type: 0 or 1, 0 if nbor doesnt exist 1 if the nbor exist
 **Author: Puneet Narula
 **Description: This function is used to send the messages to the neighbors.
 **It takes the dimension(0 for x, 1 for y, 2 for z) in which the message 
 **is to be sent and the side (0 for -ve and 1 for +ve) and the message that
 **is to be sent as argument for the function. It computes the index of the 
 **neighbor and sends the given message to the neighbor if the neighbor exists.
 **It returns 1 if the neighbor exists and 0 if the neighbor doesnot
 *********************************************************************/

int Cell :: sendInDimension(int dim,int side, NeighborMsg *msg)
{
  /** dim side   comments
   *   0   0      send left
   *   0   1      send right
   *   1   0      send up
   *   1   1      send down
   **/

  BitVec nborvec; //BitVec of neighbor
  unsigned int mask = 0;
  unsigned int mult = 1;
  int i,k;
  nborvec = myIndex;	
  nborvec.numbits = myIndex.numbits;
  i = myIndex.numbits;
  switch(side) {
  case 0:{ // equivalent to sending left in x dimension
    //or sending up in y dimension    
    // determine the bitvec for my left or up  neighbor 
    // if the neighbor doesnt exist then return 0
    if(myIndex.vec[dim] == 0) { delete msg; return 0;}
    nborvec.vec[dim] = myIndex.vec[dim] -1; 
    break;
  }
  case 1: {
    i = i/dimension;
    
    for(k=i; k>0; k--) {
      mask += mult;
      mult *= 2;
    }
    
    //if the neighbor doesnt exist then return 0
    if((myIndex.vec[dim] & mask) == mask) {
      delete msg;
      return 0;
    }
    else
      nborvec.vec[dim] = myIndex.vec[dim] +1;
    break;
  }
  }
  DEBUGN(("Nbor:sending message to x %dy %d z %d numbits %d \n",
	  nborvec.vec[0],
	  nborvec.vec[1],nborvec.vec[2],
	  nborvec.numbits));
  
  if (!msg)
    CkPrintf("sendInDimension:Neighbor message Pointer NULL\n");
  
  CkArrayIndexBitVec index(nborvec);
  arrayProxy[index].neighbor_data(msg);
  return 1;
}

/*********************************************************************
 **Method Class: Cell
 **Method Name:check_queue 
 **Arguments: None
 **Return Type: None
 **Author: Puneet Narula
 **Description: Checks if there are any out of order messages in the 
 **queue. if there are then it removes the messages for the current 
 **iteration from the queue
 *********************************************************************/
void Cell::check_queue()
{
  
  if(!FIFO_Empty(msg_queue)) {
    temp_queue= FIFO_Create();
    while(!FIFO_Empty(msg_queue)) {
      NeighborMsg *temp_msg;
      DEBUGN(("Check Queue %d\n",run_until));
      FIFO_DeQueue(msg_queue,(void **)&temp_msg);

      DEBUGN(("Check Queue x%d y%d bit%d\n",myIndex.vec[0],myIndex.vec[1],myIndex.numbits));
      if(temp_msg->run_until > run_until) {
	FIFO_EnQueue(temp_queue, temp_msg);
      }
      else if(temp_msg->run_until == run_until){
	CkArrayIndexBitVec index(myIndex);
	arrayProxy[index].neighbor_data(temp_msg);
      }
      else
	CkError("Old Message in the queue\n");
    }
    FIFO_Destroy(msg_queue);
    msg_queue = temp_queue;
  }
}

/*********************************************************************
 **Method Class: Cell
 **Method Name: neighbor_data
 **Arguments: Pointer to NeighborMsg
 **Return Type: None
 **Author: Puneet Narula
 **Description: It is an entry method which receives the data from the 
 **neighbors. It checks if the message recvd is for the current iteration.
 **If it is then it sends it to the User Data to be stored else if the
 **message is for iteration higher than the current one than the message
 **is enqueued.
 **If all the messages from the neighbors have been recieved then this 
 **method invokes the routine to do the computation on the data else it
 ** waits to receive all the required messages. Once the computation is
 **it completes it starts the next iteraion
 *********************************************************************/

void Cell :: neighbor_data(NeighborMsg *msg)
{
  
  int neighbor_side = msg->which_neighbor;
  
  if(type == 'l') {
    //check to see if the message is for the current iteration
    if(msg->run_until == run_until) {
      DEBUGN(("Recv Msg from x %d y %d z %d bits %d side %d--myIndex x %d y %d z %d bits %d- rununtil %d nborsrep %d nbors %d\n",
	      msg->nborIdx.vec[0],msg->nborIdx.vec[1],msg->nborIdx.vec[2],
	      msg->nborIdx.numbits,msg->which_neighbor,myIndex.vec[0],
	      myIndex.vec[1],myIndex.vec[2],myIndex.numbits,run_until,neighbors_reported,num_neighbors));
      reg_nbor_msg(neighbor_side, msg);
    }
    //check to see if the message is for a higher iteration then the current
    else if(msg->run_until > run_until) {
      FIFO_EnQueue(msg_queue,(void *)msg);
      DEBUGN(("Enqueueing ------------------>\n"));
      return;
    }
    else {
      //if th message recieved is for iteration already completed then it means there
      //is an error
      CkError("Message out of order %d %d x %d y %d z %d  bits %d which %d from x %d y %d z %d bits %d: neighbor data\n",
	      msg->run_until,run_until,
	      myIndex.vec[0],myIndex.vec[1],
	      myIndex.vec[2],
	      myIndex.numbits,
	      msg->which_neighbor,msg->nborIdx.vec[0],
	      msg->nborIdx.vec[1],msg->nborIdx.vec[2],
	      msg->nborIdx.numbits);
      return;
    }
    //if all the neighbors have reported then proceed to do the computation
    if (neighbors_reported == num_neighbors && num_neighbors != 0) {
      // do computation
      userData->doComputation();
      
      neighbors_reported = 0;
      for(int i=0;i<2*dimension;i++)
	nborRecvMsgCount[i] = 0;
      
      //check to see if synchronisation is requested in this iteration
      if(run_until == synchstep) {
	CkArrayIndexBitVec index(parent);
	arrayProxy[index].synchronise(new _RedMsg(0));
      }
      else {
	DEBUGA(("Calling doIteration from neighborData\n"));
	doIterations();
      }
    }
  }
  else if (type == 'v') {
    //I am virtual leaf then forward the message to the parent
    if (dimension == 1) {
      msg->numbits -= dimension;
    }
    CkArrayIndexBitVec index(parent);
    arrayProxy[index].neighbor_data(msg);
  }  
  else {
    //If I am a node then forward the message to appropriate 
    //children after splitting the message
    msg->numbits += dimension;
    forwardSplitMsg(msg ,neighbor_side);
  }
}

/*********************************************************************
 **Method Class: Cell
 **Method Name: doIterations
 **Arguments: None
 **Return Type: None
 **Author: Puneet Narula
 **Description:This method starts the iterations and starts by communicating
 **the data to all its neighbors.
 *********************************************************************/

void Cell :: doIterations() {
  /* does the following things:
     (1) check if we reach the final time step. If so then stop.
     (2) send data to four neighbors. */
  
  int i;
  num_neighbors = 0;
  justRefined = 0;

  //intialize the neighbors
  for(i=0;i<2*dimension;i++)
    neighbors[i] = 0;
  int *size ;
  size = new int [2*dimension];
  for(int i=0;i<2*dimension;i++)
    size[i] = 0;
  
  if (++run_until > run_done) {
    DEBUGR(("Finished %d my x %d y %d  z %d bits %d\n", run_until,
	    myIndex.vec[0],
	    myIndex.vec[1],
	    myIndex.vec[2],
	    myIndex.numbits));
    
    CkArrayIndexBitVec index(parent);
    arrayProxy[index].synchronise(new _RedMsg(2));
    delete []size;
    return;
  }

  int nborDim = 0;
 
  void **nborMsgDataArray = userData->getNborMsgArray(size);
  
  for(i = 0; i < 2*dimension; i++) {
    int nborDir = (i+1)% 2;
    NeighborMsg *nborMsg = new NeighborMsg;
    nborMsg->which_neighbor = i;
    nborMsg->numbits = myIndex.numbits;
    nborMsg->run_until = run_until;
    nborMsg->dataSize = size[i];
    nborMsg->nborIdx = myIndex;
    nborMsg->data = nborMsgDataArray[i];
    neighbors[nborDim*2 + nborDir] = sendInDimension(nborDim,nborDir,nborMsg);
    num_neighbors += neighbors[nborDim*2 + nborDir];
    if(nborDir == 0)
      nborDim++;
  }
  delete []nborMsgDataArray;
  delete []size;
  DEBUGN(("x = %d, y = %d z = %d numbits %d...synchstep %d ..rununtil %d neighbors %d\n",
	  myIndex.vec[0],
	  myIndex.vec[1],
	  myIndex.vec[2],
	  myIndex.numbits,
	  synchstep,
	  run_until,
	  num_neighbors));
  
  if(num_neighbors == 0) {
    CkPrintf("Finished\n");
    CkArrayIndexBitVec index(parent);
    arrayProxy[index].synchronise(new _RedMsg(0));
    return;
  }
  check_queue();
}


/* 
*************************************
Cell class methods for refinement
*************************************
*/

/*********************************************************************
 **Method Class: Cell
 **Method Name: resume
 **Arguments: Pointer to _DMsg(Dummy Message)
 **Return Type: None
 **Author: Puneet Narula
 **Description: This method is invoked to resume computation after refinement
 *********************************************************************/
void Cell :: resume(_DMsg *msg)
{
  delete msg;
  if(type == 'l') {
    DEBUGA(("Calling doIterations from resume\n"));
    doIterations();
  }
}

/*********************************************************************
 **Method Class: Cell
 **Method Name: synchronise
 **Arguments: Pointer to _RedMsg
 **Return Type: None
 **Author: Puneet Narula
 **Description:This method is invoked inorder to do a global synchronisation.
 **It waits for a message from all the children before sending the message to
 **the parent.If the current node is a root of the tree then the message is 
 **sent to synnchronise method of AmrCoordinator
 *********************************************************************/

void Cell :: synchronise(_RedMsg *msg)
{
  
  if(++synchleavrep == powerOfTwo(dimension)) {
    DEBUGS(("Cell sychronise --leaves reported %d dimension %d--my x %d my y %d my z %d bits %d \n",
	    synchleavrep,dimension,
	    myIndex.vec[0],myIndex.vec[1],myIndex.vec[2],myIndex.numbits));
    synchleavrep = 0;
    if(type == 'n') {
      CkArrayIndexBitVec index(parent);
      arrayProxy[index].synchronise(msg);
    }
    else if(type == 'r') {
      DEBUGS(("Reporting synchronisation message to coordinator x %d y %d z %d bits %d reduction msg type %d\n"
	      ,myIndex.vec[0],myIndex.vec[1],
	      myIndex.vec[2],myIndex.numbits,msg->type));
      CProxy_AmrCoordinator coordProxy(coordHandle);
      coordProxy.synchronise(msg);
    }
    else 
      CkError("Error in sychronisation step\n");
  }
  else
    delete msg;
}

/*********************************************************************
 **Method Class: Cell
 **Method Name: refineExec
 **Arguments: Pointer to _DMsg (Dummy Message)
 **Return Type: None
 **Author: Puneet Narula
 **Description:This entry method is invoked after the synchronisation to determine 
 **if a leaf needs refinement. Refinement criterion is implemented by the user.
 *********************************************************************/
void Cell :: refineExec(_DMsg *msg)
{
  delete msg;

  if(type == 'l' && justRefined == 0) {
    //refine Criterion is a function that has to be implemented by the user
    if(userData->refineCriterion()) {
      //send a regular refine message
      DEBUGRC(("Refining node x %d y %d z %d bits %d\n",
	       myIndex.vec[0],myIndex.vec[1],myIndex.vec[2],
	       myIndex.numbits));
      synchstep += synchinterval;
      refine(new _RefineMsg(0));
    }
    else {
      //if the cell doesnot satisfy the refinement criterion then
      // just inform the parent that the cell is done for now
      synchstep += synchinterval;
      DEBUGRC(("refineExec: x= %d y = %d z %d numbits %d synchstep %d\n",
	       myIndex.vec[0], 
	       myIndex.vec[1],
	       myIndex.vec[2],
	       myIndex.numbits,
	       synchstep));
      CkArrayIndexBitVec index(parent);
      arrayProxy[index].synchronise(new _RedMsg(1));
    }
  }
  else if(type =='n' || type == 'r')
      synchstep += synchinterval;
}

/*********************************************************************
 **Method Class: Cell
 **Method Name: refine
 **Arguments: Pointer to _RefineMsg
 **Return Type: None
 **Author: Puneet Narula
 **Description:This entry method is invoked if the leaf node satisfies 
 **the refinement criteria or it needs to be refined to maintain the 
 **invariant in the tree that every leaf node has its neighboring leaf
 **nodes atmost 1 level away from it. It refines the leaf and makes the 
 **current cell a node.
 *********************************************************************/
void Cell :: refine(_RefineMsg *msg)
{
  // this member function is called when leaf needs 
  //to be refined further
  int i, j,flag;

  if(autorefine == 0) {
    autorefine = msg->autorefine;
    flag = 0;
    // check if the message is an autorefine message
    if( msg->autorefine == 1)
      retidx = msg->index;
  }
  else
    flag = 1;

  //only a leaf can be refined.....
  if( type == 'l') {
    if (neighbors_reported == 0) {
      DEBUGRC(("Refine leaf: x %d y %d z %d numbits %d synchstep %d..autorefine %d\n",
	       myIndex.vec[0],myIndex.vec[1],
	       myIndex.vec[2],myIndex.numbits,
	       synchstep,autorefine));
      int size=0;
      
      void **cmsgData = userData->fragmentForRefine(&size);
      ChildInitMsg ***cmsg = new ChildInitMsg** [dimension];
      int temp=2;
      for(int index = 0; index < dimension;index++){
	temp = ((index==0)? 2:powerOfTwo(index));
	cmsg[index] = new ChildInitMsg* [temp];
      }
      for (int u =0; u<dimension;u++) {
	temp = ((u ==0)? 2:powerOfTwo(u));
	for(int v=0; v<temp ; v++) {
	  cmsg[u][v] = new ChildInitMsg;
	  //send message to the virtual leaves who are my children
	  // to convert themselves to leaf and inturn create virtual leaves
	  if(cmsgData[u*2+v] == NULL)
	    CkError("Error: FragmentForRefine() didnot give a message to be sent to %d child\n"
		    ,u*2 + v);
	  else {

	    cmsg[u][v]->run_until =  run_until;
	    cmsg[u][v]->num_neighbors = 0;
	    
	    if(synchstep == run_until)
	      cmsg[u][v]->synchstep = synchstep+synchinterval;
	    else 
	      cmsg[u][v]->synchstep = synchstep;
	    
	    cmsg[u][v]->dataSize = size;
	    cmsg[u][v]->data = cmsgData[u*2+v];
	    
		
	    // Send msg to children to change themselves to leaves 
	    CkArrayIndexBitVec aindex(children[u][v]);
	    arrayProxy[aindex].change_to_leaf(cmsg[u][v]);
	  }
	}//End of inner for loop with v as index
      } //End of outer for loop with u as the index
      
      /**************************************************
       *delete the child msg array 
       **************************************************/
      /*  for (int u=0;u<dimension;u++)
	delete[] cmsg[u];
	delete[] cmsg;*/
      delete []cmsgData;
      /*send message to neighbors that i am refining*/
      //autorefine code
      num_neighbors = 0;
      for(i=0; i <dimension; i++) 
	for(j=0; j<2; j++) 
	  num_neighbors += sendInDimension(i,j);
      
      //make yourself a node as the leaf has been refined
      type = 'n';
      if(statCollection) {
	CProxy_StatCollector grpProxy(gid);
	StatCollector* grp = grpProxy.ckLocalBranch();
	if(autorefine == 0)
	  grp->incrementRefine();
	else
	  grp->incrementAutorefine();
      }
    }//End if with neighbors reported variable condition
    else {
      CkError("Neighbors reported non zero--even after synchronisation in refine x %d y %d z %d bits %d run_until %d parent x %d y %d z %d bits %d\n",
		 myIndex.vec[0],myIndex.vec[1],myIndex.vec[2],myIndex.numbits,run_until,
	      parent.vec[0],parent.vec[1],parent.vec[2],parent.numbits);
    }
  }//End if with type = 'l' as the condition
  else if (autorefine == 1) {
    //A node or root can recieve an autorefine message but not a regular refine message
    DEBUGRC(("Refine non leaf but autorefine 1 \n"));
    DEBUGRC(("Refine Ready....autorefine 1\n"));
    if (flag == 0)
      autorefine = 0;
    refineReady(msg->index,1);
    if(statCollection) {
      CProxy_StatCollector grpProxy(gid);
      StatCollector* grp = grpProxy.ckLocalBranch();
      grp->incrementAutorefine();
    }
  }
  else {
    //error if non leaf receives a normal refine message
    CkPrintf("Error:Autorefine %d",autorefine);
    CkError(" Refine Msg received by a non leaf--type %c\n",type); 
  }
}

/*********************************************************************
 **Method Class: Cell
 **Method Name: change_to_leaf
 **Arguments: Pointer to ChildInitMsg
 **Return Type: None
 **Author: Puneet Narula
 **Description: This entry method is invoked by refine on virtual leaves.
 **which need to be converted to new leaves to add a new level.
 *********************************************************************/

void Cell :: change_to_leaf(ChildInitMsg *msg)
{
  int i;
  //This function can be called only on virtual leaves
  
  if(type == 'v') {
    //If I am currently a virtual leaf then change my type to 
    // leaf and create virtual leaves as my children
    justRefined = 1;
    run_until = msg->run_until;
    num_neighbors = msg->num_neighbors;
    //num_neighbors = 0;
    synchstep = msg->synchstep;
    userData = AmrUserData::createDataWrapper(myIndex,dimension,msg->data,msg->dataSize);
    type = 'l';
    delete msg;
    int temp = powerOfTwo(dimension);
    _ArrInitMsg **cmsg = new _ArrInitMsg* [temp];
    
    for(i=0; i<powerOfTwo(dimension) ;i++) {
      cmsg[i] = new _ArrInitMsg;
      cmsg[i]->parent = myIndex;
      cmsg[i]->type = 'v';
      cmsg[i]->interval = synchinterval;
      cmsg[i]->totalIterations = run_done;
      cmsg[i]->coordHandle = coordHandle;
      cmsg[i]->depth = -1;
      cmsg[i]->statCollection = statCollection;
      cmsg[i]->gid = gid;
    }
    create_children(cmsg);
    if(statCollection){
      CProxy_StatCollector grpProxy(gid);
      StatCollector *grp = grpProxy.ckLocalBranch();
      grp->registerMe();
    }
    CkArrayIndexBitVec index(parent);
    arrayProxy[index].refine_confirmed(new _DMsg(myIndex, 0));
  }
  else 
    CkError("change to leaf called on a node or leaf\n");
}

/*********************************************************************
 **Method Class: Cell
 **Method Name: refine_confirmed
 **Arguments: Pointer to _DMsg(Dummy Message)
 **Return Type: None
 **Author: Puneet Narula
 **Description:Each virtual leaf after changing itself to leaf invokes
 **this method on the parent to confirm that it has refined. Also all
 **the neighbors of the refined leaf invokes this method to confirm to 
 **the neighbor that it is at an appropriate level to receive messages
 **from its neighbors children which are leaves now.
 *********************************************************************/

void Cell :: refine_confirmed(_DMsg *msg)
{
  
  DEBUGRC(("Refine confirmed for x %d y %d z %d bits %d-- sender x %d y %d z %d numbits %d from pos %d--- refined reports %d, total rep %d\n",myIndex.vec[0],
	   myIndex.vec[1],myIndex.vec[2],myIndex.numbits,msg->sender.vec[0],
	   msg->sender.vec[1],msg->sender.vec[2],msg->sender.numbits,msg->from,
	   refined,(2*dimension+num_neighbors)));
  delete msg;
  int temp = powerOfTwo(dimension);
  if(++refined == (temp+num_neighbors)) {
    
    /*auto refinement completed send message to neighbor*/
    if(autorefine == 1) {
      DEBUGRC(("getting refine confirms bcoz of autorefine\n"));
      DEBUGRC(("Refine Ready....autorefine 2\n"));
      refineReady(retidx,2);
      autorefine = 0;
      num_neighbors =0;
      neighbors_reported = 0;
    }
    else {
      /*regular refine completed*/
      CkArrayIndexBitVec index(parent);
      arrayProxy[index].synchronise(new _RedMsg(1));
      num_neighbors = 0;
      neighbors_reported = 0;
    }
    
    refined = 0;
  }
}

/*********************************************************************
 **Method Class: Cell
 **Method Name: sendInDimension
 **Arguments: dimension to send(integer), side(+ or -)
 **Return Type: 0 or 1, 0 if nbor doesnt exist 1 if the nbor exist
 **Author: Puneet Narula
 **Description: This function is used to send the messages to the neighbors
 **to check if the neighbor is at an appropriate refinement level.
 **It takes the dimension(0 for x, 1 for y, 2 for z) in which the message 
 **is to be sent and the side (0 for -ve and 1 for +ve) and sends a message
 **to the neighbor to check its refinement level. It computes the index of the 
 **neighbor and sends _RefineChk  message to the neighbor if the neighbor exists.
 **It returns 1 if the neighbor exists and 0 if the neighbor doesnot
 *********************************************************************/
int Cell :: sendInDimension(int dim,int side)
{
  /** dim side   comments
   *   0   0      send left
   *   0   1      send right
   *   1   0      send up
   *   1   1      send down
   **/
  
  BitVec nborvec; //bitvec of neighbor
  unsigned int mask = 0;
  unsigned int mult = 1;
  int i,k;
  nborvec = myIndex;	
  i = nborvec.numbits = myIndex.numbits;
  switch(side) {
  case 0: // equivalent to sending left in x dimension
          //or sending up in y dimension
    
    // determine the bitvec for my left or up  neighbor  
   
    // if the neighbor doesnt exist then return 0
    if(myIndex.vec[dim] == 0)  return 0;
    
    nborvec.vec[dim] = myIndex.vec[dim] - 1;
    break;
  case 1:
    i = i/dimension;
    
    for(k=i; k>0; k--) {
      mask += mult;
      mult *= 2;
    }
    
    //if the neighbor doesnt exist then return 0
    if((myIndex.vec[dim] & mask)==mask) return 0;//check for the boundary
    nborvec.vec[dim] = myIndex.vec[dim] + 1;
    break;
  }

   DEBUGN(("sending message to x %dy %d z %d numbits %d \n",nborvec.vec[0],
    nborvec.vec[1],nborvec.vec[2],nborvec.numbits));
  CkArrayIndexBitVec index(nborvec);
  arrayProxy[index].checkRefine(new _RefineChkMsg(myIndex,run_until));
  return 1;
}


/*********************************************************************
 **Method Class: Cell
 **Method Name: checkRefine
 **Arguments: Pointer to _RefineChkMsg
 **Return Type: None
 **Author: Puneet Narula
 **Description:This entry method is invoked by neighboring leaf which
 **is refining to make sure that the neighbor is at an appropriate
 **refinement level.If the node receiving this message is a virtual
 **leaf then it needs to refine itself to maintain the tree invariant
 **else it just confirms to the neighbor that it is at an appropriate 
 **level to recieve messages from the neghbors children
 *********************************************************************/
void Cell :: checkRefine(_RefineChkMsg* msg) 
{
  if(type == 'v') {
   
    CkArrayIndexBitVec index(parent);
    arrayProxy[index].refine(new _RefineMsg(1,msg->index));
    delete msg;
  }
  else {
    /*if(synchstep == msg->run_until && (type == 'l' || type == 'n'))
      synchstep +=synchinterval;*/
    DEBUGRC(("Refine Ready....autorefine 3\n"));
    refineReady(msg->index,3);
    delete msg;
  }
}

/*********************************************************************
 **Method Class: Cell
 **Method Name: refineReady
 **Arguments: index of the neighbor who initiated the query(bitvec),
 **Return Type: None
 **Author: Puneet Narula
 **Description: This method just confirms to the neighbor that it is 
 **at an appropriate level to receive message from its(neighbors) children
 **after refinement is completed
 *********************************************************************/
void Cell :: refineReady(BitVec retid,int pos)
{

  CkArrayIndexBitVec index(retid);
  arrayProxy[index].refine_confirmed(new _DMsg(myIndex, pos));
}

void Cell :: pup(PUP::er &p)
{
  DEBUGN(("Puping the array\n")); 
  ArrayElementT<BitVec>::pup(p);
  p(dimension);
  p(type);

  p|userData;
  
  parent.pup(p);
  //children array
  if (p.isUnpacking()) {
    children = new BitVec*[dimension];
    for(int i=0; i< dimension; i++) {
      int size = (i==0)? 2: powerOfTwo(i); 
      children[i] = new BitVec[size];
    }
  }

  for(int i=0; i < dimension;i++) {
    int size = ((i==0)? 2:powerOfTwo(i));
    p((char*)children[i], size*sizeof(BitVec));
  }
  
  myIndex.pup(p);
  p(num_neighbors);
  p(neighbors_reported);
  p(run_until);
  p(run_done);
  /* Check for the kind of pupper being called
    if(p.isPacking()){ 
    CkPrintf("Packing:Pup being called x %d y %d z %d bits %d--runUntil %d pe %d\n",
    myIndex.vec[0],myIndex.vec[1],myIndex.vec[2],
    myIndex.numbits,run_until,CkMyPe());
    }
    else if(p.isUnpacking()) {
    CkPrintf("UnPacking:Pup being called x %d y %d z %d bits %d--runUntil %d pe %d\n",
    myIndex.vec[0],myIndex.vec[1],myIndex.vec[2],
    myIndex.numbits,run_until,CkMyPe());
    }
    else {
    CkPrintf("Sizing:Pup being called x %d y %d z %d bits %d--runUntil %d pe %d\n",
    myIndex.vec[0],myIndex.vec[1],myIndex.vec[2],
    myIndex.numbits,run_until,CkMyPe());
    }
  */
  p(justRefined);
  
  if (p.isUnpacking()) {
    neighbors = new int[2*dimension];
    nborRecvMsgCount = new int[2*dimension];
    for(int i=0; i< 2*dimension; i++)
      nborRecvMsgCount[i] = 0;
  }
  p(neighbors,2*dimension);
#ifdef CHK_PT_PUP
  p(nborRecvMsgCount,2*dimension);
#endif

  int count = 0;
  int temp = powerOfTwo(dimension -1);
  //neighborRecv Msg buffer
  if(p.isUnpacking()) {
    nborRecvMsgBuff = new NeighborMsg** [2*dimension];
    for(int i=0;i<2*dimension;i++) {
      nborRecvMsgBuff[i] = new NeighborMsg* [temp];
      for(int j=0; j<temp;j++)
	nborRecvMsgBuff[i][j] = NULL;
    }
  }
  else{
    for(int i=0;i<2*dimension;i++){
      for(int j=0;j<temp;j++){
	if(nborRecvMsgBuff && nborRecvMsgBuff[i] && nborRecvMsgBuff[i][j]){
	  count++;
#ifndef CHK_PT_PUP
	  NeighborMsg* tempMsg = new NeighborMsg;
	  cpyNborMsg(tempMsg,nborRecvMsgBuff[i][j]);
	  delete nborRecvMsgBuff[i][j];
	  nborRecvMsgBuff[i][j] = NULL;
	  CkPrintf("Resending message from the partial messages %d \n",count);
	  arrayProxy[myIndex].neighbor_data(tempMsg);
#endif
	}
      }
    }
  }
#ifdef CHK_PT_PUP  
  p(count);
  if(count >0){
    for(int i=0;i<2*dimension;i++){
      for(int j=0;j<temp;j++){
	if(nborRecvMsgBuff && nborRecvMsgBuff[i] && nborRecvMsgBuff[i][j])
	  nborRecvMsgBuff[i][j]->pup(p);
      }
    }
  }
#endif
 
  //queue stuff
#ifdef CHK_PT_PUP
  if(!p.isUnpacking()){
    msg_count = FIFO_Fill(msg_queue);
    p(msg_count);
    if(msg_count >0) {
      for(int i=0; i<msg_count; i++) {
	NeighborMsg *temp_msg;
	FIFO_DeQueue(msg_queue, (void **) &temp_msg);
	temp_msg->pup(p);
	FIFO_EnQueue(msg_queue,(void*)temp_msg);
      }
    }
  }
  else{
    msg_queue = FIFO_Create();
    p(msg_count);
    if(msg_count > 0) {
      while(msg_count >0) {
	NeighborMsg *temp_msg = new NeighborMsg;
	temp_msg->pup(p);
	FIFO_EnQueue(msg_queue, temp_msg);
	msg_count--;
      }
    } 
  }
#endif

#ifndef CHK_PT_PUP
  if(!p.isUnpacking()){
    while(!FIFO_Empty(msg_queue)) {
      NeighborMsg* temp_msg;
      FIFO_DeQueue(msg_queue,(void**) &temp_msg);
      arrayProxy[myIndex].neighbor_data(temp_msg);
    }
  }
  else
    msg_queue = FIFO_Create();
#endif
  
  p(refined);
  p(autorefine);
  retidx.pup(p);
  p(synchleavrep);
  p(synchinterval);
  p(synchstep);
  
  //coordhandle
  p|coordHandle;
  p|statCollection;
  p|gid;
  if(p.isPacking() && statCollection && type == 'l')
  {
    CProxy_StatCollector grpProxy(gid);
    StatCollector* grp = grpProxy.ckLocalBranch();
    grp->migrating();
  }
  else if(p.isUnpacking() && statCollection && type == 'l')
  {	
    CProxy_StatCollector grpProxy(gid);
    StatCollector* grp = grpProxy.ckLocalBranch();
    grp->registerMe();
  }
  
}


void Cell :: cpyNborMsg(NeighborMsg* dest,NeighborMsg* src)
{
  
  dest->which_neighbor = src->which_neighbor;
  dest->run_until = src->run_until;
  dest->numbits = src->numbits;
  dest->nborIdx  = src->nborIdx;
  dest->dataSize = src->dataSize;
  dest->data = (void *) new char[src->dataSize];
  memcpy(dest->data,src->data,src->dataSize);
}



/*
*************************************
Cell2D class methods
*************************************
*/

/*********************************************************************
 **Method Class: Cell2D
 **Method Name: Cell2D(Constructor)
 **Arguments: Pointer to _ArrInitMsg
 **Return Type: None
 **Author: Puneet Narula
 **Description:Constructor for Cell2D class. Initializes the data members
 **for the super class(Cell) and starts the iterations.
 *********************************************************************/
Cell2D :: Cell2D(_ArrInitMsg *msg ) 
{
  int i, j, k;
  //  "Cell Constructor for 2D Tree

  dimension = 2;
  CProxy_Cell2D aProxy(thisArrayID);
  arrayProxy = *(CProxy_Cell*) &aProxy;
  
  init_cell(msg);

  //This function does useful work only during the initial creation
  //of the tree before the first iteration
  treeSetup(msg);

  
  delete msg;
 
  // only leaves have to participate in the computation
  if(type == 'l') {
    userData = AmrUserData::createDataWrapper(myIndex,dimension);
    doIterations();
  }
  else
    userData = NULL;
}

/*********************************************************************
 **Method Class: Cell2D
 **Method Name: create_children
 **Arguments: Pointer to an Array of _ArrinitMsg
 **Return Type: None
 **Author: Puneet Narula
 **Description:This method creates new children for the node and takes
 **the messages to be sent to the newly created children as the argument
 **to the function.
 *********************************************************************/

void Cell2D :: create_children(_ArrInitMsg** cmsg) 
{
  int i,j,k;
  CProxy_Cell2D aProxy = *(CProxy_Cell2D *) &arrayProxy;
 
 //Determine who my children are
  for(i =0; i<dimension;i++) {
    for(j=0; j<2;j++) {
      for(k =0; k<dimension ; k++)
	children[i][j].vec[k] = 2*myIndex.vec[k];
      children[i][j].numbits = myIndex.numbits + dimension;
    }
  }
  children[0][1].vec[0] += 1;
  children[1][0].vec[1] += 1;
  children[1][1].vec[0] += 1;
  children[1][1].vec[1] += 1;
  
  
  //create all my children which can either be leaves or nodes
  j = 0;
  for(i= 0; i<dimension ; i++) 
    for(k=0; k<2; k++) {
      //DEBUGR(("Inserting %d %d %d bits %d mybits %d\n",children[i][k].vec[0],children[i][k].vec[1],children[i][k].vec[2],children[i][k].numbits,myIndex.numbits));
      CkArrayIndexBitVec index(children[i][k]);
      aProxy[index].insert(cmsg[j++]);
      
    }
  aProxy.doneInserting();
  // deleting he message array after the messages have been sent
  delete [] cmsg;
}


/*********************************************************************
 **Method Class: Cell2D
 **Method Name: 
 **Arguments: 
 **Return Type: None
 **Author: Puneet Narula
 **Description:
 *********************************************************************/

void Cell2D :: reg_nbor_msg(int neighbor_side, NeighborMsg *msg)
{
  int i;
   
  if(msg->numbits > myIndex.numbits) {
    // the message received is smaller than required for this leaf's
    //grain size--wait for the other message if necessary
    if(nborRecvMsgCount && nborRecvMsgCount[neighbor_side] == 0){
      nborRecvMsgCount[neighbor_side]++;
      nborRecvMsgBuff[neighbor_side][0] = msg;
    }
    else if(nborRecvMsgCount && nborRecvMsgCount[neighbor_side] == 1){
      nborRecvMsgCount[neighbor_side]++;
      nborRecvMsgBuff[neighbor_side][1] = msg;
      userData->combineAndStore(nborRecvMsgBuff[neighbor_side][0],nborRecvMsgBuff[neighbor_side][1]);
      delete nborRecvMsgBuff[neighbor_side][0];
      delete nborRecvMsgBuff[neighbor_side][1];
      neighbors_reported++;
    }
    else {
      CkError("Error in the message received from %d neighbor :my_x %d my_y %d myZ %d\n",
	      neighbor_side, myIndex.vec[0], myIndex.vec[1],myIndex.vec[2]);
    }
  } 
  else if (msg->numbits == myIndex.numbits) {
    if(nborRecvMsgCount && nborRecvMsgCount[neighbor_side] == 0){
      nborRecvMsgCount[neighbor_side] +=2;
      neighbors_reported++;
      DEBUG(("Calling Store: x %d y %d z %dnobits %d nborside %d\n",
	      myIndex.vec[0], myIndex.vec[1],myIndex.vec[2],myIndex.numbits,
	      neighbor_side));
      userData->store(msg);
      delete msg;
    }
    else
      CkError("Wrong size message received from %d neighbor :my_x %d my_y %d my_z %d mybits %d msgbits %d\n"
	      , neighbor_side, myIndex.vec[0],myIndex.vec[1],myIndex.vec[2],
	      myIndex.numbits,msg->numbits);
  }
  else
    CkError("Bigger Message received then my grainsize from %d neighbor :my_x %d my_y %d my_z %d numbits %d msg numbits %d\n",
	    neighbor_side,
	    myIndex.vec[0],
	    myIndex.vec[1],
	    myIndex.vec[2],
	    myIndex.numbits,
	    msg->numbits); 
}



void Cell2D :: forwardSplitMsg(NeighborMsg *msg ,int neighbor_side)
{
   //if neighbor data msg is received by node or root
  switch(neighbor_side) {
  case  NEG_Y: 
  //message to be forwarded to child with index 0,0
  //                           child with index 1,0 
   frag_msg(msg, 0, 0, 0,1);
   break;
  case POS_Y: 
  //message to be forwarded to child with index 0,1
  //                           child with index 1,1 
   frag_msg(msg, 1,0,1,1);
   break;
  case NEG_X: 
  //message to be forwarded to child with index 0,0
  //                           child with index 0,1 
   frag_msg(msg,0,0,1,0);
   break;
  case POS_X: 
      //message to be forwarded to child with index 1,0
      //                           child with index 1,1 
   frag_msg(msg,0,1,1,1);
   break;
    
  }
}

void Cell2D :: frag_msg(NeighborMsg *msg, int child1_x, int child1_y,
		      int child2_x, int child2_y)
{

  int i;
  NeighborMsg ** splitMsgArray = userData->fragment(msg,2);
  //  NeighborMsg ** splitMsgArray = AmrUserData::fragment(msg,2);
  for(i=0;i<2;i++){
    if(splitMsgArray[i]){
      splitMsgArray[i]->which_neighbor =  msg->which_neighbor;
      splitMsgArray[i]->run_until = msg->run_until;
      splitMsgArray[i]->numbits = msg->numbits;
      splitMsgArray[i]->nborIdx = msg->nborIdx;
    }
    else
      CkError("Error: 2 messages were not recieved by frag_msg from fragment\n");
  }
  

  CkArrayIndexBitVec index1(children[child1_x][child1_y]);
  arrayProxy[index1].neighbor_data(splitMsgArray[0]);
  CkArrayIndexBitVec index2(children[child2_x][child2_y]);
  arrayProxy[index2].neighbor_data(splitMsgArray[1]);
  delete msg;
}

/*
*************************************
Cell3D class methods
*************************************
*/

Cell3D :: Cell3D(_ArrInitMsg *msg ) 
{
  int i, j, k;
  //  "Cell Constructor for 3D Tree
  DEBUGT(("Cell 3D constructor \n"));
  dimension = 3;
  CProxy_Cell3D aProxy(thisArrayID);
  arrayProxy = *(CProxy_Cell*) &aProxy;
  
  init_cell(msg);
  DEBUGT(("Init Msg done \n"));
  //This function does useful work only during the initial creation
  //of the tree before the first iteration
  treeSetup(msg);
  DEBUGT(("Tree SetUp done\n"));
  
  delete msg;
 
  DEBUGT(("Deleted the message\n"));
  // only leaves have to participate in the computation
  if(type == 'l') {
    userData = AmrUserData::createDataWrapper(myIndex,dimension);
    doIterations();
  }
  else 
    userData = NULL;
}

void Cell3D :: create_children(_ArrInitMsg** cmsg) 
{
  int i,j,k;
  CProxy_Cell3D aProxy = *(CProxy_Cell3D *) &arrayProxy;
  //Determine who my children are
  for(i =0; i<dimension;i++) {
    int temp = powerOfTwo(i);
    if (i==0)
      temp = 2;
    for(j=0; j<temp;j++) {
      for(k =0; k<dimension ; k++)
	children[i][j].vec[k] = 2*myIndex.vec[k];
      children[i][j].numbits = myIndex.numbits + dimension;
    }
  }
  children[0][1].vec[0] += 0x0001;
  children[1][0].vec[1] += 0x0001;
  children[1][1].vec[0] += 0x0001;
  children[1][1].vec[1] += 0x0001;

  children[2][0].vec[2] += 0x0001;

  children[2][1].vec[2] += 0x0001;
  children[2][1].vec[0] += 0x0001;

  children[2][2].vec[2] += 0x0001;
  children[2][2].vec[1] += 0x0001;

  children[2][3].vec[2] += 0x0001;
  children[2][3].vec[0] += 0x0001;
  children[2][3].vec[1] += 0x0001;

  
  //create all my children which can either be leaves or nodes
  j = 0;
  for(i= 0; i<dimension ; i++) {
    int size = powerOfTwo(i);
    if (i==0)
      size = 2;
    for(k=0; k<size; k++) {
      //DEBUGR(("Inserting %d %d %d bits %d mybits %d\n",children[i][k].vec[0],children[i][k].vec[1],children[i][k].vec[2],children[i][k].numbits,myIndex.numbits));
      CkArrayIndexBitVec index(children[i][k]);
      if(cmsg[j])
	DEBUGT(("Child msg is good \n"));
      else
	DEBUGT(("There is a problem dude\n"));
      aProxy[index].insert(cmsg[j++]);
      
    }
  }
  aProxy.doneInserting();
  // deleting he message array after the messages have been sent
  delete [] cmsg;
}


void Cell3D :: forwardSplitMsg(NeighborMsg *msg ,int neighbor_side)
{
   //if neighbor data msg is received by node or root
  switch(neighbor_side) {
  case  NEG_Y: 
    //message to be forwarded to child with index 0,0
    //                           child with index 1,0 
    //                           child with index 2,0
    //                           child with index 2,1
    //   frag_msg(msg, 0, 0, 1,0,2,0,2,1);
    frag_msg(msg, 0, 0, 0,1,2,0,2,1);
   break;
  case POS_Y: 
    //message to be forwarded to child with index 0,1
    //                           child with index 1,1 
    //                           child with index 2,2
    //                           child with index 2,3
    // frag_msg(msg, 0,1,1,1,2,2,2,3);
    frag_msg(msg, 1,0,1,1,2,2,2,3);
   break;
  case NEG_X: 
    //message to be forwarded to child with index 0,0
    //                           child with index 0,1 
    //                           child with index 2,0 
    //                           child with index 2,2 
   frag_msg(msg,0,0,1,0,2,0,2,2);
   break;
  case POS_X: 
    //message to be forwarded to child with index 1,0
    //                           child with index 1,1 
    //                           child with index 2,1
    //                           child with index 2,3
   frag_msg(msg,0,1,1,1,2,1,2,3);
   break;
  case NEG_Z:
    //message to be forwarded to child with index 0,0
    //                           child with index 0,1 
    //                           child with index 1,0
    //                           child with index 1,1
    frag_msg(msg,0,0,0,1,1,0,1,1);
    break;
  case POS_Z:
    //message to be forwarded to child with index 2,0
    //                           child with index 2,1 
    //                           child with index 2,2
    //                           child with index 2,3
    frag_msg(msg,2,0,2,1,2,2,2,3);
    break;
  }
}

void Cell3D :: frag_msg(NeighborMsg *msg, int child1_x, int child1_y,
			int child2_x, int child2_y, int child3_x,
			int child3_y, int child4_x, int child4_y)
{
  int i;
  NeighborMsg ** splitMsgArray = userData->fragment(msg,4);
  //NeighborMsg ** splitMsgArray = AmrUserData::fragment(msg,4);
  for(i=0;i<4;i++){
    if(splitMsgArray[i]){
      splitMsgArray[i]->which_neighbor =  msg->which_neighbor;
      splitMsgArray[i]->run_until = msg->run_until;
      splitMsgArray[i]->numbits = msg->numbits;
      splitMsgArray[i]->nborIdx = msg->nborIdx;
    }
    else
      CkError("Error: 4 messages were not recieved by frag_msg from fragment\n");
  }
  
  CkArrayIndexBitVec index1(children[child1_x][child1_y]);
  arrayProxy[index1].neighbor_data(splitMsgArray[0]);
  CkArrayIndexBitVec index2(children[child2_x][child2_y]);
  arrayProxy[index2].neighbor_data(splitMsgArray[1]);
  CkArrayIndexBitVec index3(children[child3_x][child3_y]);
  arrayProxy[index3].neighbor_data(splitMsgArray[2]);
  CkArrayIndexBitVec index4(children[child4_x][child4_y]);
  arrayProxy[index4].neighbor_data(splitMsgArray[3]);
  delete msg;
}

void Cell3D :: reg_nbor_msg(int neighbor_side, NeighborMsg *msg)
{
  int i;
  
  if(msg->numbits > myIndex.numbits) {
    // the message received is smaller than required for this leaf's
    //grain size--wait for the other message if necessary
    if(nborRecvMsgCount && nborRecvMsgCount[neighbor_side] < 3){
      nborRecvMsgBuff[neighbor_side][nborRecvMsgCount[neighbor_side]] = msg;
      nborRecvMsgCount[neighbor_side]++;
    }
    else if(nborRecvMsgCount && nborRecvMsgCount[neighbor_side] == 3){
      nborRecvMsgCount[neighbor_side]++;
      nborRecvMsgBuff[neighbor_side][3] = msg;
      userData->combineAndStore(nborRecvMsgBuff[neighbor_side][0],nborRecvMsgBuff[neighbor_side][1],nborRecvMsgBuff[neighbor_side][2],nborRecvMsgBuff[neighbor_side][3]);
      delete nborRecvMsgBuff[neighbor_side][0];
      delete nborRecvMsgBuff[neighbor_side][1];
      delete nborRecvMsgBuff[neighbor_side][2];
      delete nborRecvMsgBuff[neighbor_side][3];
      for(int j =0; j<4;j++)
	nborRecvMsgBuff[neighbor_side][j] = NULL;
      neighbors_reported++;
    }
    else {
      CkError("Error in the message received from %d neighbor :my_x %d my_y %d my_z %d\n",
	      neighbor_side, myIndex.vec[0], myIndex.vec[1],myIndex.vec[2]);
    }
  } 
  else if (msg->numbits == myIndex.numbits) {
    if(nborRecvMsgCount){
      if(nborRecvMsgCount[neighbor_side] == 0){
	nborRecvMsgCount[neighbor_side] +=4;
	neighbors_reported++;
	DEBUGJ(("Calling Store: x %d y %d z %d nobits %d nborside %d\n",
		myIndex.vec[0], myIndex.vec[1],myIndex.vec[2],
		myIndex.numbits, neighbor_side));
	userData->store(msg);
	delete msg;
      }
      else
	CkError("Receiving the message from the same neighbor twice x %d y %d \
z %d bits %d nborSide %d:nbor x %d y %d z %d bits %d count %d\n",
		myIndex.vec[0],myIndex.vec[1],myIndex.vec[2],myIndex.numbits,
		neighbor_side,msg->nborIdx.vec[0],msg->nborIdx.vec[1],
		msg->nborIdx.vec[2],msg->nborIdx.numbits,nborRecvMsgCount[neighbor_side]);
    }
    else
      CkError("NborRecvPoint Null Pointer for nborSide  %d :my_x %d my_y %d my_z %d mybits %d msgbits %d count %d\n"
	      ,neighbor_side, myIndex.vec[0],myIndex.vec[1],
	      myIndex.vec[2],myIndex.numbits,msg->numbits, 
	      nborRecvMsgCount[neighbor_side]);
  }
  else
    CkError("Bigger Message received then my grainsize from %d neighbor :my_x %d my_y %d my_z %d numbits %d msg numbits %d\n",
	    neighbor_side,
	    myIndex.vec[0],
	    myIndex.vec[1],
	    myIndex.vec[2],
	    myIndex.numbits,
	    msg->numbits); 
}

/*
*************************************
Cell1D class methods
*************************************
*/

Cell1D :: Cell1D(_ArrInitMsg *msg ) 
{
  int i, j, k;
  //  "Cell Constructor for 2D Tree
  DEBUGT(("Cell 1D constructor \n"));
  dimension = 1;
  CProxy_Cell1D aProxy(thisArrayID);
  arrayProxy = *(CProxy_Cell*) &aProxy;
  
  init_cell(msg);
  DEBUGT(("Init Msg done \n"));
  //This function does useful work only during the initial creation
  //of the tree before the first iteration
  treeSetup(msg);
  DEBUGT(("Tree SetUp done\n"));
  
  delete msg;
 
  DEBUGT(("Deleted the message\n"));
  // only leaves have to participate in the computation
  if(type == 'l') {
    //   AmrUserData tempData;
    //   userData = tempData.createData();
    userData = AmrUserData::createDataWrapper(myIndex,dimension);
    DEBUGA(("Calling doIteration from the constructor\n"));
    doIterations();
    
  }
  else
    userData = NULL;
}

void Cell1D :: create_children(_ArrInitMsg** cmsg) 
{
  int i,j,k;
  CProxy_Cell1D aProxy = *(CProxy_Cell1D *) &arrayProxy;
  //Determine who my children are
  for(i =0; i<dimension;i++) {
    for(j=0; j<2;j++) {
      for(k =0; k<dimension ; k++)
	children[i][j].vec[k] = 2*myIndex.vec[k];
      children[i][j].numbits = myIndex.numbits + dimension;
    }
  }
  children[0][1].vec[0] += 0x0001;
  
  //create all my children which can either be leaves or nodes
  j = 0;
  for(i= 0; i<dimension ; i++) 
    for(k=0; k<2; k++) {
      //DEBUGR(("Inserting %d %d %d bits %d mybits %d\n",children[i][k].vec[0],children[i][k].vec[1],children[i][k].vec[2],children[i][k].numbits,myIndex.numbits));
      CkArrayIndexBitVec index(children[i][k]);
      if(cmsg[j])
	DEBUG(("Child msg is good \n"));
      else
	CkError(("There is a problem dude\n"));
      aProxy[index].insert(cmsg[j++]);
      
    }
  aProxy.doneInserting();
  // deleting he message array after the messages have been sent
  delete [] cmsg;
}


void Cell1D :: reg_nbor_msg(int neighbor_side, NeighborMsg *msg)
{
  int i;
   
  if(msg->numbits > myIndex.numbits) {
    // the message received is smaller than required for this leaf's
    //grain size--wait for the other message if necessary
    CkError("Error: reg_nbor_msg-- should not have differernt numbits:msg bits %d , mybits %d\n",msg->numbits,myIndex.numbits);
  }
  else if (msg->numbits == myIndex.numbits) {
    if(nborRecvMsgCount && nborRecvMsgCount[neighbor_side] == 0){
      nborRecvMsgCount[neighbor_side] +=2;
      neighbors_reported++;
      DEBUG(("Calling Store: x %d y %d z %dnobits %d nborside %d\n",
	     myIndex.vec[0], myIndex.vec[1],myIndex.vec[2],myIndex.numbits,
	     neighbor_side));
      userData->store(msg);
      delete msg;
    }
    else
      CkError("Wrong size message received from %d neighbor :my_x %d my_y %d my_z %d mybits %d msgbits %d\n"
	      , neighbor_side, myIndex.vec[0],myIndex.vec[1],myIndex.vec[2],
	      myIndex.numbits,msg->numbits);
  }
  else
    CkError("Bigger Message received then my grainsize from %d neighbor :my_x %d my_y %d my_z %d numbits %d msg numbits %d\n",
	    neighbor_side,
	    myIndex.vec[0],
	    myIndex.vec[1],
	    myIndex.vec[2],
	    myIndex.numbits,
	    msg->numbits); 
}

void Cell1D :: forwardSplitMsg(NeighborMsg *msg ,int neighbor_side)
{
  //if neighbor data msg is received by node or root then it 
  //needs to forwarded but need not be split for the 1D case
  
  switch (neighbor_side) {
  case NEG_X: {
    //if the message is from my left neighbor
    //then forward the message to my left child
    CkArrayIndexBitVec index1(children[0][0]);
    arrayProxy[index1].neighbor_data(msg);
    break;
  }
  case POS_X: {
    //if the message is from my right neighbor 
    //then forward the message to my right child
    CkArrayIndexBitVec index2(children[0][1]);
    arrayProxy[index2].neighbor_data(msg);
    break;
  }
  }
  
}


/************************
AmrUserData methods
************************/

NeighborMsg ** AmrUserData :: fragment(NeighborMsg *msg,int nMsg)
{
  NeighborMsg **msgArray = new NeighborMsg* [nMsg];
  int size = msg->dataSize;
  void **msgArrayData = fragmentNborData(msg->data, &size);
  
  for(int i=0;i<nMsg;i++) {
    msgArray[i] = new NeighborMsg;
    msgArray[i]->dataSize = size;
    msgArray[i]->data = msgArrayData[i];
  }
  return msgArray;
}

void AmrUserData :: combineAndStore(NeighborMsg* msg1, NeighborMsg *msg2)
{
  //should be a dynamic array but it will work till the 4D
  void *data[2];
  data[0] = msg1->data;
  data[1] = msg2->data;
  if(msg1->dataSize == msg2->dataSize)
    combineAndStore(data, msg1->dataSize,msg1->which_neighbor);
  else
    CkError("Error: AmrUserData::combineAndStore messages are of differentSizes\n");

}

void AmrUserData :: combineAndStore(NeighborMsg* msg1, NeighborMsg *msg2,NeighborMsg *msg3, NeighborMsg *msg4)
{
  //should be a dynamic array but it will work till the 4D
  void* data[4];
  data[0] = msg1->data;
  data[1] = msg2->data;
  data[2] = msg3->data;
  data[3] = msg4->data;
  combineAndStore(data, msg1->dataSize,msg1->which_neighbor);
  // else
  //  CkError("Error: AmrUserData::combineAndStore messages are of differentSizes\n");

}

void AmrUserData :: store(NeighborMsg *msg)
{
  store(msg->data, msg->dataSize, msg->which_neighbor);
}



#include "amr.def.h"
