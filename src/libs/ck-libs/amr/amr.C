#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "amr.h"


AmrCoordinator :: AmrCoordinator(_DMsg* msg)
{
  delete msg;
  arrayProxy = CProxy_Cell2D::ckNew();
  synchInterval = 50;
  depth = 2;
  dimension = 2;
  totalIterations = 500;
  myHandle = thishandle;
  create_tree();
}

AmrCoordinator :: AmrCoordinator(StartUpMsg *msg)
{
  //CProxy_Cell2D arrayProxy;
  synchInterval = msg->synchInterval;
  depth = msg->depth;
  dimension = msg->dimension;
  totalIterations = msg->totalIterations;
  delete msg;
  switch (dimension){
  case 1:
    break;
  case 2:
    arrayProxy =  CProxy_Cell2D::ckNew();
    break;
  case 3:
    break;
  }
  myHandle = thishandle;
  create_tree();
}

void AmrCoordinator :: synchronise(_RedMsg *msg)
{
  if(msg->type == 0){
    //ready for refine
    delete msg;
    arrayProxy.refineExec(new _DMsg);
  }
  else if (msg->type == 1) { //refine completed
    delete msg;
    DEBUGS(("Broadcasting a message to resume\n"));
    arrayProxy.resume(new _DMsg);
  }
  else if (msg->type == 2) { //end of all iterations
    CkPrintf("Completed iterations\n");
    CkPrintf("Size of bitvector %d\n",sizeof(bitvec));
    delete msg;
    CkExit();
  }

}


void AmrCoordinator :: create_tree()
{
  bitvec root;
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
  DEBUGR(("Inserting %d %d bits %d\n",root.vec[0],root.vec[1],root.numbits));
  CkArrayIndexBitVec index(root);
  switch(dimension) {
  case 1: {
    CProxy_Cell1D aProxy = (CProxy_Cell1D) arrayProxy;
    aProxy[index].insert(msg);
    aProxy.doneInserting();
    break;
  }
  case 2: {
    CProxy_Cell2D aProxy = (CProxy_Cell2D) arrayProxy;
    aProxy[index].insert(msg);
    aProxy.doneInserting();
    break;
  }
  }
 
  
}

/*
***************************************
*NeighborMsg methods
***************************************
*/
void* NeighborMsg :: pack(NeighborMsg *msg)
{
  int bufSize = msg->dataSize;
  bufSize += 4*sizeof(int);
  char *buf = (char *) CkAllocBuffer(msg,bufSize);
  memcpy(buf,&(msg->which_neighbor),sizeof(int));
  buf += sizeof(int);
  memcpy(buf,&(msg->run_until),sizeof(int));
  buf += sizeof(int);
  memcpy(buf,&(msg->numbits),sizeof(int));
  buf += sizeof(int);
  memcpy(buf,&(msg->dataSize),sizeof(int));
  buf += sizeof(int);
  memcpy(buf,msg->data,msg->dataSize);
  // if (!msg->copyToSerialBuffer(buf))
  //  CkError("Error: Pack()--Unable to copy the user data NeighborMsg to serial buffer\n");
  buf -= 4*sizeof(int);
  delete msg;
  return (void *) buf;
}

NeighborMsg* NeighborMsg :: unpack(void* inbuf)
{
  //should get the message already allocated from  the derived class i.e
  //user data to be communicated to the user
  char *buf = (char *) inbuf;
  NeighborMsg *msg = (NeighborMsg *) CkAllocBuffer(inbuf,sizeof(NeighborMsg));
  
  memcpy(&(msg->which_neighbor),buf,sizeof(int));
  buf += sizeof(int);
  memcpy(&(msg->run_until),buf,sizeof(int));
  buf += sizeof(int);
  memcpy(&(msg->numbits),buf,sizeof(int));
  buf += sizeof(int);
  memcpy(&(msg->dataSize),buf,sizeof(int));
  buf += sizeof(int);
  msg->data = (void *)new char[msg->dataSize];
  memcpy(msg->data,buf,msg->dataSize);
  //  if(!(msg->copySerialBufToMsg(buf)))
  // CkError("Error: unpack()--Unable to copy the serial buffer to user data NeighborMsg\n");
  CkFreeMsg(inbuf);
  return msg;
}


/*
***************************************
*ChildInitMsg methods
***************************************
*/
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

ChildInitMsg* ChildInitMsg :: unpack(void* inbuf)
{
  //should get the message already allocated from  the derived class i.e
  //user data to be communicated to the user
  char *buf = (char *) inbuf;
  ChildInitMsg *msg = (ChildInitMsg *)  CkAllocBuffer(inbuf,sizeof(ChildInitMsg));
  
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

  //  if(!(msg->copySerialBufToMsg(buf)))
  // CkError("Error: unpack()--Unable to copy the serial buffer to user data NeighborMsg\n");
  CkFreeMsg(inbuf);
  return msg;
}
/*
*****************************************
*Cell Class Methods
*****************************************
*/

void Cell :: init_cell(_ArrInitMsg *msg)
{
  int i;

  //Initailze the fields in the array element of class Cell
  type = msg->type; 
  parent = msg->parent;
  
  synchinterval  = msg->interval;
  synchstep = synchinterval;
  synchleavrep = 0;
  //number of iterations for the program
  run_done = msg->totalIterations;    
  

  myIndex = thisIndex;	//get the bytes for my index
  //DEBUGR(("My index x %d y %d bits %d\n",myIndex.vec[0],myIndex.vec[1],myIndex.numbits));
  num_neighbors = 0;
  neighbors_reported = 0;
  
  coordHandle = msg->coordHandle;
  //msg_count = low_count = up_count = right_count = left_count = 0;
  nborRecvMsgCount = new int [2*dimension];
  for(i=0;i<2*dimension;i++)
    nborRecvMsgCount[i] = 0;

  nborRecvMsgBuff = new NeighborMsg** [2*dimension];
  for(i=0;i<2*dimension;i++)
    nborRecvMsgBuff[i] = new NeighborMsg* [2];
  
  run_until = 0;
  refined = 0;
  autorefine = 0;
  
  msg_queue = FIFO_Create();
  start_ptr = NULL;
  //create children array
  children = new bitvec* [dimension];
  for (i = 0; i < dimension; i++) 
    children[i] = new bitvec [2];
  //neighbors array
  neighbors = new int [2*dimension];
}

void Cell :: treeSetup(_ArrInitMsg *msg)
{
  if((type == 'r' || type == 'n') && msg->depth > 0) {
    // root or node with depth > 0
    //This Part is executed only during the initial creation of the tree
    
    //send messages to create nodes further 
    _ArrInitMsg **cmsg;
    cmsg = new _ArrInitMsg* [2*dimension];
    for(int i=0; i<dimension *2 ;i++) {
      cmsg[i] = new _ArrInitMsg;
      cmsg[i]->parent = myIndex;
      cmsg[i]->type = 'n';
      cmsg[i]->interval = synchinterval;
      cmsg[i]->depth  = (msg->depth) - 1;
      cmsg[i]->totalIterations = run_done;
      cmsg[i]->coordHandle  = coordHandle;
      //cmsg[i]->cell_sz = cell_sz;
    }
    DEBUGT(("Before create children\n"));
    create_children(cmsg);
  } 
  else if(type == 'n' && msg->depth == 0) {
    // If I am a node and the depth is zero
    // that is I am actually a leaf so change my type to leaf
    // and create virtual leaves as my children
    // this part is also executed during the intial creation of the tree
    
    //create messages to be sent to my children who are virtual leaves
    _ArrInitMsg **cmsg;
    cmsg = new _ArrInitMsg* [2*dimension];
    for(int i=0; i<dimension *2 ;i++) {
      cmsg[i] = new _ArrInitMsg;
      cmsg[i]->parent = myIndex;
      cmsg[i]->type = 'v';
      cmsg[i]->interval = synchinterval;
      cmsg[i]->depth  = (msg->depth) - 1;
      cmsg[i]->totalIterations = run_done;
      cmsg[i]->coordHandle  = coordHandle;
      //cmsg[i]->cell_sz = cell_sz/2 + 1;
    }
    
    create_children(cmsg);
    //change my type to leaf
    type = 'l';
  }

}

int Cell :: sendInDimension(int dim,int side, NeighborMsg *msg)
{
  /** dim side   comments
   *   0   0      send left
   *   0   1      send right
   *   1   0      send up
   *   1   1      send down
   **/

  bitvec nborvec; //bitvec of neighbor
  unsigned short mask = 0;
  unsigned short mult = 1;
  int i,k;
  nborvec.vec[dim] = myIndex.vec[dim];	
  i = nborvec.numbits = myIndex.numbits;
  switch(side) {
  case 0: // equivalent to sending left in x dimension
          //or sending up in y dimension    
    // determine the bitvec for my left or up  neighbor  
    
    // if the neighbor doesnt exist then return 0
    if(myIndex.vec[dimension-dim-1] == 0)  return 0;

    nborvec.vec[dimension - dim -1] = myIndex.vec[dimension -dim-1] - 1;
    break;
  case 1:
    i = i/dimension;
    
    for(k=i; k>0; k--) {
      mask += mult;
      mult *= 2;
    }
    
    //if the neighbor doesnt exist then return 0
    if((myIndex.vec[dimension-dim-1] & mask)==mask) return 0;//check for the boundary
    nborvec.vec[dimension -dim -1] = myIndex.vec[dimension - dim -1] + 1;
    break;
  }
  DEBUGT(("sending message to x %dy %d z %d numbits %d \n",nborvec.vec[0],
	  nborvec.vec[1],nborvec.vec[2],nborvec.numbits));
  if (!msg)
    CkPrintf("Dude...thats bad\n");
  //CProxy_Cell arr(thisArrayID);
  CkArrayIndexBitVec index(nborvec);
  arrayProxy[index].neighbor_data(msg);
  return 1;
}

void Cell :: reg_nbor_msg(int neighbor_side, NeighborMsg *msg)
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
      CkError("Error in the message received from %d neighbor :my_x %d my_y %d \n",
	      neighbor_side, myIndex.vec[0], myIndex.vec[1]);
    }
  } 
  else if (msg->numbits == myIndex.numbits) {
   if(nborRecvMsgCount && nborRecvMsgCount[neighbor_side] == 0){
     nborRecvMsgCount[neighbor_side] +=2;
     neighbors_reported++;
     DEBUGJ(("Calling Store: x %d y %d nobits %d nborside %d\n",
	     myIndex.vec[0], myIndex.vec[1],myIndex.numbits,
	     neighbor_side));
     userData->store(msg);
     delete msg;
   }
   else
     CkError("Wrong size message received from %d neighbor :my_x %d my_y %d mybits %d msgbits %d\n"
	     , neighbor_side, myIndex.vec[0],myIndex.vec[1],myIndex.numbits,msg->numbits);
  }
  else
    CkError("Bigger Message received then my grainsize from %d neighbor :my_x %d my_y %d numbits %d msg numbits %d\n",
	    neighbor_side,
	    myIndex.vec[0],
	    myIndex.vec[1],
	    myIndex.numbits,
	    msg->numbits); 
}

void Cell :: neighbor_data(NeighborMsg *msg)
{
  
  int neighbor_side = msg->which_neighbor;
  
  if(type == 'l') {
    if(msg->run_until == run_until) {
      reg_nbor_msg(neighbor_side, msg);
    }
    else if(msg->run_until > run_until) {
      FIFO_EnQueue(msg_queue,(void *)msg);
      //msg_count++;
    }
    else {
      CkError("Message out of order %d %d x %d y %d  bits %d which %d: neighbor data\n",
	      msg->run_until,run_until,
	      myIndex.vec[0],myIndex.vec[1],
	      myIndex.numbits,
	      msg->which_neighbor);
    }
    if (neighbors_reported == num_neighbors) {
      // do computation
      userData->doComputation();
      
      neighbors_reported = 0;
      for(int i=0;i<2*dimension;i++)
	nborRecvMsgCount[i] = 0;
      
      if(run_until == synchstep) {
	//CProxy_Cell synchproxy(thisArrayID);
	CkArrayIndexBitVec index(parent);
	arrayProxy[index].synchronise(new _RedMsg(0));
      }
      else
	doIterations();
    }
  }
  else if (type == 'v') {
    //I am virtual leaf myx %d myy %d:neighbor data\n",
    if (dimension == 1) {
      msg->numbits -= dimension;
    }
    CkArrayIndexBitVec index(parent);
    arrayProxy[index].neighbor_data(msg);
  }  
  else {
    msg->numbits += dimension;
    forwardSplitMsg(msg ,neighbor_side);
  }
}

void Cell::check_queue()
{
  
  if(!FIFO_Empty(msg_queue)) {
    temp_queue= FIFO_Create();
    while(!FIFO_Empty(msg_queue)) {
      NeighborMsg *temp_msg;
      DEBUGR(("Check Queue %d\n",run_until));
      FIFO_DeQueue(msg_queue,(void **)&temp_msg);
      if(temp_msg->run_until > run_until) {
	FIFO_EnQueue(temp_queue, temp_msg);
      }
      else if(temp_msg->run_until== run_until){
	//msg_count--;
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


/* 
*************************************
Cell class methods for refinement
*************************************
*/
void Cell :: resume(_DMsg *msg)
{
  delete msg;
  if(type == 'l')
    doIterations();
}

void Cell :: synchronise(_RedMsg *msg)
{
  
  if(++synchleavrep == 2*dimension) {
    DEBUGS(("Cell sychronise --leaves reported %d dimension %d--my x %d my y %d bits %d \n",synchleavrep,dimension,
	     myIndex.vec[0],myIndex.vec[1],myIndex.numbits));
    synchleavrep = 0;
    _RedMsg *cmsg = new _RedMsg;
    memcpy(cmsg,msg,sizeof(_RedMsg));
    delete msg;
    if(type == 'n') {
      // memcpy(cmsg,msg,sizeof(_RedMsg));
      //CProxy_Cell synchproxy(thisArrayID);
      CkArrayIndexBitVec index(parent);
      arrayProxy[index].synchronise(cmsg);
    }
    else if(type == 'r') {
      DEBUGS(("Reporting synchronisation message to coordinator x %d y %d bits %d reduction msg type %d\n",myIndex.vec[0],myIndex.vec[1],myIndex.numbits,cmsg->type));
      CProxy_AmrCoordinator coordProxy(coordHandle);
      coordProxy.synchronise(cmsg);
    }
    else 
      CkError("Error in sychronisation step\n");
  }
  else
    delete msg;
}

void Cell :: refineExec(_DMsg *msg)
{
  delete msg;
  if(type == 'l') {
    //refine Criterion is a function that has to be implemented by the user
    if(userData->refineCriterion()) {
      //send a regular refine message
      refine(new _RefineMsg(0));
    }
    else {
      //if the cell doesnot satisfy the refinement criterion then
      // just inform the parent that the cell is done for now
      synchstep += synchinterval;
      DEBUGRC(("refineExec: x= %d y = %d numbits %d synchstep %d\n",
	       myIndex.vec[0], 
	       myIndex.vec[1],
	       myIndex.numbits,
	       synchstep));
      //CProxy_Cell cproxy(thisArrayID);
      CkArrayIndexBitVec index(parent);
      arrayProxy[index].synchronise(new _RedMsg(1));
    }
  }
}

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
  //  delete msg;
  }
  else
    flag = 1;

  //only a leaf can be refined.....
  if( type == 'l') {
    if (neighbors_reported == 0) {
      DEBUGRC(("Refine leaf: x %d y %d numbits %d synchstep %d..autorefine %d\n",
	       myIndex.vec[0],myIndex.vec[1],myIndex.numbits,
	       synchstep,autorefine));
      int size=0;
      
      void **cmsgData = userData->fragmentForRefine(&size);
      ChildInitMsg ***cmsg = new ChildInitMsg** [dimension];
      for(int index = 0; index < dimension;index++)
	cmsg[index] = new ChildInitMsg* [2];
      DEBUGT(("Done Creating Child Init messages\n"));
      for (int u =0; u<dimension;u++) {
	for(int v=0; v<2 ; v++) {
	  cmsg[u][v] = new ChildInitMsg;
	  //send message to the virtual leaves who are my children
	  // to convert themselves to leaf and inturn create virtual leaves
	  if(cmsgData[u*2+v] == NULL)
	    CkError("Error: FragmentForRefine() didnot give a message to be sent to %d child\n"
		    ,u*2 + v);
	  else {

	    cmsg[u][ v]->run_until =  run_until;

	    cmsg[u][v]->num_neighbors = 0;
	    
	    if(synchstep == run_until)
	      cmsg[u][v]->synchstep = synchstep+synchinterval;
	     else 
	       cmsg[u][v]->synchstep = synchstep;

	    cmsg[u][v]->dataSize = size;
	    cmsg[u][v]->data = cmsgData[u*2+v];
	    
	    // Send msg to children to change themselves to leaves 
	    CkArrayIndexBitVec index(children[u][v]);
	    arrayProxy[index].change_to_leaf(cmsg[u][v]);
	  }
	}//End of inner for loop with v as index
      } //End of outer for loop with u as the index

      /**************************************************
       *should delete the child msg array here
       **************************************************/

     /*send message to neighbors that i am refining*/
     //autorefine code
     num_neighbors = 0;
     for(i=0; i <dimension; i++) 
       for(j=0; j<2; j++) 
	 num_neighbors += sendInDimension(i,j);
     
     //make yourself a node as the leaf has been refined
     type = 'n';
    }//End if with neighbors reported variable condition
    else CkError("Neighbors reported non zero--even after synchronisation in refine\n");
  }//End if with type = 'l' as the condition
  else if (autorefine == 1) {
    //A node or root can recieve an autorefine message but not a regular refine message
    DEBUGRC(("Refine non leaf but autorefine 1 \n"));
    DEBUGRC(("Refine Ready....autorefine 1\n"));
    if (flag == 0)
      autorefine = 0;
    refineReady(msg->index,1);
  }
  else {
    //error if non leaf receives a normal refine message
    CkPrintf("Error:Autorefine %d",autorefine);
    CkError(" Refine Msg received by a non leaf--type %c\n",type); 
  }
}

void Cell :: change_to_leaf(ChildInitMsg *msg)
{
  int i;
  //This function can be called only on virtual leaves
  
  if(type == 'v') {
    //If I am currently a virtual leaf then change my type to 
    // leaf and create virtual leaves as my children

    run_until = msg->run_until;
    num_neighbors = msg->num_neighbors;
    synchstep = msg->synchstep;
    //    AmrUserData tempData;
    // userData = tempData.createData(msg->data,msg->dataSize);
    
    userData = AmrUserData::createData(msg->data,msg->dataSize);
    type = 'l';
    delete msg;
    
    _ArrInitMsg **cmsg = new _ArrInitMsg* [dimension*2];
    
    for(i=0; i<dimension *2 ;i++) {
      cmsg[i] = new _ArrInitMsg;
      cmsg[i]->parent = myIndex;
      cmsg[i]->type = 'v';
      cmsg[i]->interval = synchinterval;
      cmsg[i]->totalIterations = run_done;
      cmsg[i]->coordHandle = coordHandle;

    }
    create_children(cmsg);

    CkArrayIndexBitVec index(parent);
    arrayProxy[index].refine_confirmed(new _DMsg(myIndex, 0));
  }
  else 
    CkError("change to leaf called on a node or leaf\n");
}

void Cell :: refine_confirmed(_DMsg *msg)
{
  
  DEBUGRC(("Refine confirmed for x %d y %d bits %d-- sender x %d y %d numbits %d from pos %d--- refined reports %d, total rep %d\n",myIndex.vec[0],
	   myIndex.vec[1],myIndex.numbits,msg->sender.vec[0],
	   msg->sender.vec[1],msg->sender.numbits, msg->from,
	   refined,(2*dimension+num_neighbors)));
  delete msg;
  if(++refined == (2*dimension+num_neighbors)) {
    //if(++refined == 2*DIMENSION) {
    
    /*auto refinement completed send message to */
    if(autorefine == 1) {
      DEBUGRC(("getting refine confirms bcoz of autorefine\n"));
      DEBUGRC(("Refine Ready....autorefine 2\n"));
      refineReady(retidx,2);
      autorefine = 0;
    }
    else {
      /*regular refine completed*/
      //CProxy_Cell cproxy(thisArrayID);
      CkArrayIndexBitVec index(parent);
      arrayProxy[index].synchronise(new _RedMsg(1));
    }
    
    refined = 0;
  }
}

int Cell :: sendInDimension(int dim,int side)
{
  /** dim side   comments
   *   0   0      send left
   *   0   1      send right
   *   1   0      send up
   *   1   1      send down
   **/
  
  bitvec nborvec; //bitvec of neighbor
  unsigned short mask = 0;
  unsigned short mult = 1;
  int i,k;
  nborvec.vec[dim] = myIndex.vec[dim];	
  i = nborvec.numbits = myIndex.numbits;
  switch(side) {
  case 0: // equivalent to sending left in x dimension
          //or sending up in y dimension
    
    // determine the bitvec for my left or up  neighbor  
   
    // if the neighbor doesnt exist then return 0
    if(myIndex.vec[dimension-dim-1] == 0)  return 0;
    
    nborvec.vec[dimension - dim -1] = myIndex.vec[dimension -dim-1] - 1;
    break;
  case 1:
    i = i/dimension;
    
    for(k=i; k>0; k--) {
      mask += mult;
      mult *= 2;
    }
    
    //if the neighbor doesnt exist then return 0
    if((myIndex.vec[dimension-dim-1] & mask)==mask) return 0;//check for the boundary
    nborvec.vec[dimension -dim -1] = myIndex.vec[dimension - dim -1] + 1;
    break;
  }
  //CProxy_Cell arr(thisArrayID);
  CkArrayIndexBitVec index(nborvec);
  arrayProxy[index].checkRefine(new _RefineChkMsg(myIndex));
  return 1;
}

void Cell :: checkRefine(_RefineChkMsg* msg) 
{
  if(type == 'v') {
   
    CkArrayIndexBitVec index(parent);
    arrayProxy[index].refine(new _RefineMsg(1,msg->index));
    delete msg;
  }
  else {
    if(synchstep == run_until && (type == 'l'|| type == 'n'))
      synchstep +=synchinterval;
    DEBUGRC(("Refine Ready....autorefine 3\n"));
    refineReady(msg->index,3);
    delete msg;
  }
}

void Cell :: refineReady(bitvec retid,int pos)
{
  //  CkPrintf("RefineReady....\n");

  CkArrayIndexBitVec index(retid);
  arrayProxy[index].refine_confirmed(new _DMsg(myIndex, pos));
}

/*
*************************************
Cell2D class methods
*************************************
*/

Cell2D :: Cell2D(_ArrInitMsg *msg ) 
{
  int i, j, k;
  //  "Cell Constructor for 2D Tree
  DEBUGT(("Cell 2D constructor \n"));
  dimension = 2;
  CProxy_Cell2D aProxy(thisArrayID);
  arrayProxy = (CProxy_Cell)aProxy;
  
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
    userData = AmrUserData::createData();
   doIterations();
  }
}

void Cell2D :: create_children(_ArrInitMsg** cmsg) 
{
  int i,j,k;
  CProxy_Cell2D aProxy = (CProxy_Cell2D) arrayProxy;
  //Determine who my children are
  for(i =0; i<dimension;i++) {
    for(j=0; j<2;j++) {
      for(k =0; k<dimension ; k++)
	children[i][j].vec[k] = 2*myIndex.vec[k];
      children[i][j].numbits = myIndex.numbits + dimension;
    }
  }
  children[0][1].vec[0] += 0x0001;
  children[1][0].vec[1] += 0x0001;
  children[1][1].vec[0] += 0x0001;
  children[1][1].vec[1] += 0x0001;
  
  //  CProxy_Cell2D arr(thisArrayID);
  
  //create all my children which can either be leaves or nodes
  j = 0;
  for(i= 0; i<dimension ; i++) 
    for(k=0; k<2; k++) {
      //DEBUGR(("Inserting %d %d bits %d mybits %d\n",children[i][k].vec[0],children[i][k].vec[1],children[i][k].numbits,myIndex.numbits));
      CkArrayIndexBitVec index(children[i][k]);
      if(cmsg[j])
	DEBUGT(("Child msg is good \n"));
      else
	DEBUGT(("There is a problem dude\n"));
      aProxy[index].insert(cmsg[j++]);
      
    }
  aProxy.doneInserting();
}

void Cell2D :: doIterations() {
  /* does the following things:
     (1) check if we reach the final time step. If so then stop.
     (2) send data to four neighbors. */
  
  int i;
  num_neighbors = 0;
  //refinement test
  //intialize the neighbors
  for(i=0;i<2*dimension;i++)
    neighbors[i] = 0;
  
  if (++run_until > run_done) {
    DEBUGR(("Finished %d my x %d y %d  bits %d\n", run_until,myIndex.vec[0],
	     myIndex.vec[1],
	     myIndex.numbits));
    
    //CProxy_Cell2D cproxy(thisArrayID);
    CkArrayIndexBitVec index(parent);
    arrayProxy[index].synchronise(new _RedMsg(2));
    return;
  }
  int nborDim = 0;
  int size = 0;
  void **nborMsgDataArray = userData->getNborMsgArray(&size);
  
  for(i = 0; i < 2*dimension; i++) {
    int nborDir = (i+1)% 2;
    NeighborMsg *nborMsg = new NeighborMsg;
    nborMsg->which_neighbor = i;
    nborMsg->numbits = myIndex.numbits;
    nborMsg->run_until = run_until;
    nborMsg->dataSize = size;
    nborMsg->data = nborMsgDataArray[i];
    neighbors[nborDim*2 + nborDir] = sendInDimension(nborDim,nborDir,nborMsg);
    num_neighbors += neighbors[nborDim*2 + nborDir];
    if(nborDir == 0)
      nborDim++;
  }
  /*if(run_until == 60 || run_until == 59) */
  DEBUGR(("x = %d, y = %d numbits %d...rundone %d ..rununtil %d neighbors %d\n",
	  myIndex.vec[0],
	  myIndex.vec[1],
	  myIndex.numbits,
	  run_done,
	  run_until,
	  num_neighbors));
  
  if(num_neighbors == 0) {
    CkPrintf("Finished\n");
    _DMsg *msg1 = new _DMsg;
    //CProxy_Cell cproxy(thisArrayID);
    CkArrayIndexBitVec index(parent);
    arrayProxy[index].synchronise(new _RedMsg(0));
    return;
  }
  check_queue();
}

void Cell2D :: forwardSplitMsg(NeighborMsg *msg ,int neighbor_side)
{
   //if neighbor data msg is received by node or root
  switch(neighbor_side) {
  case  NEG_Y: 
  //message to be forwarded to child with index 0,0
  //                           child with index 1,0 
   frag_msg(msg, 0, 0, 1,0);
   break;
  case POS_Y: 
  //message to be forwarded to child with index 0,1
  //                           child with index 1,1 
   frag_msg(msg, 0,1,1,1);
   break;
  case NEG_X: 
  //message to be forwarded to child with index 0,0
  //                           child with index 0,1 
   frag_msg(msg,0,0,0,1);
   break;
  case POS_X: 
      //message to be forwarded to child with index 1,0
      //                           child with index 1,1 
   frag_msg(msg,1,0,1,1);
   break;
    
  }
}

void Cell2D :: frag_msg(NeighborMsg *msg, int child1_x, int child1_y,
		      int child2_x, int child2_y)
{
  //int new_cell_sz = msg->cell_sz/2 +1;
  int i;
  NeighborMsg ** splitMsgArray = userData->fragment(msg);

  for(i=0;i<2;i++){
    if(splitMsgArray[i]){
      splitMsgArray[i]->which_neighbor =  msg->which_neighbor;
      splitMsgArray[i]->run_until = msg->run_until;
      splitMsgArray[i]->numbits = msg->numbits;
    }
    else
      CkError("Error: 2 messages were not recieved by frag_msg from fragment\n");
  }
  
  //  CProxy_Cell arr(thisArrayID);
  CkArrayIndexBitVec index1(children[child1_x][child1_y]);
  arrayProxy[index1].neighbor_data(splitMsgArray[0]);
  CkArrayIndexBitVec index2(children[child2_x][child2_y]);
  arrayProxy[index2].neighbor_data(splitMsgArray[1]);
  delete msg;
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
  arrayProxy = (CProxy_Cell)aProxy;
  
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
    userData = AmrUserData::createData();
    DEBUGT(("Created the userData\n"));
   doIterations();
    
  }
}

void Cell1D :: create_children(_ArrInitMsg** cmsg) 
{
  int i,j,k;
  CProxy_Cell1D aProxy = (CProxy_Cell1D) arrayProxy;
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
      //DEBUGR(("Inserting %d %d bits %d mybits %d\n",children[i][k].vec[0],children[i][k].vec[1],children[i][k].numbits,myIndex.numbits));
      CkArrayIndexBitVec index(children[i][k]);
      if(cmsg[j])
	DEBUGT(("Child msg is good \n"));
      else
	DEBUGT(("There is a problem dude\n"));
      aProxy[index].insert(cmsg[j++]);
      
    }
  aProxy.doneInserting();
}

void Cell1D :: doIterations() {
  /* does the following things:
     (1) check if we reach the final time step. If so then stop.
     (2) send data to four neighbors. */
  
  int i;
  num_neighbors = 0;
  //refinement test
  //intialize the neighbors
  for(i=0;i<2*dimension;i++)
    neighbors[i] = 0;
  DEBUGT(("In do Iterations for x %d y %d z %d bits %d\n",myIndex.vec[0],
	  myIndex.vec[1],myIndex.vec[2],
	  myIndex.numbits));
  if (++run_until > run_done) {
    DEBUGR(("Finished %d my x %d y %d  bits %d\n", run_until,myIndex.vec[0],
	     myIndex.vec[1],
	     myIndex.numbits));
    
    CkArrayIndexBitVec index(parent);
    arrayProxy[index].synchronise(new _RedMsg(2));
    return;
  }
  int nborDim = 0;
  int size = 0;
  void **nborMsgDataArray = userData->getNborMsgArray(&size);

  for(i = 0; i < 2*dimension; i++) {
    int nborDir = (i+1)% 2;
    NeighborMsg *nborMsg = new NeighborMsg;
    nborMsg->which_neighbor = i;
    nborMsg->numbits = myIndex.numbits;
    nborMsg->run_until = run_until;
    nborMsg->dataSize = size;

    nborMsg->data = nborMsgDataArray[i];
    DEBUGJ(("Sending the message dim %d dir %d i %d\n",nborDim , nborDir, i));
    neighbors[nborDim*2 + nborDir] = sendInDimension(nborDim,nborDir,nborMsg);
    num_neighbors += neighbors[nborDim*2 + nborDir];
    if(nborDir == 0)
      nborDim++;
  }

  DEBUGR(("x = %d, y = %d numbits %d...rundone %d ..rununtil %d neighbors %d\n",
	  myIndex.vec[0],
	  myIndex.vec[1],
	  myIndex.numbits,
	  run_done,
	  run_until,
	  num_neighbors));
  
  if(num_neighbors == 0) {
    CkPrintf("Finished\n");
    _DMsg *msg1 = new _DMsg;
    //CProxy_Cell cproxy(thisArrayID);
    CkArrayIndexBitVec index(parent);
    arrayProxy[index].synchronise(new _RedMsg(0));
    return;
  }
  check_queue();
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

NeighborMsg ** AmrUserData :: fragment(NeighborMsg *msg)
{
  NeighborMsg **msgArray = new NeighborMsg* [2];
  int size = msg->dataSize;
  void **msgArrayData = fragmentNborData(msg->data, &size);
  
  for(int i=0;i<2;i++) {
    msgArray[i] = new NeighborMsg;
    msgArray[i]->dataSize = size;
    msgArray[i]->data = msgArrayData[i];
  }
  return msgArray;
}

void AmrUserData :: combineAndStore(NeighborMsg* msg1, NeighborMsg *msg2)
{
  if(msg1->dataSize == msg2->dataSize)
    combineAndStore(msg1->data,msg2->data, msg1->dataSize,msg1->which_neighbor);
  else
    CkError("Error: AmrUserData::combineAndStore messages are of differentSizes\n");

}

void AmrUserData :: store(NeighborMsg *msg)
{
  store(msg->data, msg->dataSize, msg->which_neighbor);
}



#include "amr.def.h"
