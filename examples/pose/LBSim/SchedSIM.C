#include "topology.C"

const int DEBUG=0;
int maxObj;

int myinitialise(QUEUE *queue)
{  
    if ((queue->front = (NODE *)malloc(sizeof(NODE))) == NULL)
    return 0;    
    queue->rear = queue->front;
    queue->front->data=-1;
    queue->front->next = NULL;
    return 1;
}

int myenqueue(QUEUE *queue, long int key)
{
    NODE *newnode;
    if ((newnode=(NODE *)malloc(sizeof(NODE))) == NULL)
        return 0;
    newnode->data = key;
    newnode->next = NULL;
    /* Add to the queue */
    queue->rear->next = newnode;
    queue->rear = newnode;
    return 1;
}

int mydequeue(QUEUE *queue)
{
    NODE *oldnode;
    long int key;
    oldnode = queue->front->next;
    key = oldnode->data;
    /* Check if removing the last node from the queue */
    if (queue->front->next->next == NULL)
        queue->rear = queue->front;
    else
        queue->front->next = queue->front->next->next;
    delete(oldnode);
    //printf("Item %d dequeued\n",key);
    return key;
}

int myisempty(QUEUE *queue)
{
    return (queue->front == queue->rear);
}

int myqlength(QUEUE *queue)
{
    NODE *ptr;
    int count = 0;
    ptr=queue->front;
    while(!(ptr==queue->rear))
    {
      count++;
      ptr=ptr->next;
    }
    return count;
}

void printqueue(QUEUE *queue)
{
    NODE *ptr;
    int count=0;
    ptr=queue->front;
    while(!(ptr==queue->rear))
    {
      count++;	
      ptr=ptr->next;
      printf("%ld:%ld  ",count,ptr->data);
    }
    printf("\n");
}

    
// schedulerObject implementation

// the non-empty constructor

schedulerObject::schedulerObject(SchedulerData *m)
{
  int i,avg=0,recvid;
  srand48(42);
  maxObjects = m->maxObjects;
  connectivity = m->connectivity;
  id = m->id;
  data=m->data;
  lbtopolen=m->lbtopolen;
  lbtopo=new char[lbtopolen];
  strcpy(lbtopo,m->lbtopo);
  delete(m);
  queue= new QUEUE;
  myinitialise(queue);
  load=0;
  WorkTime=0;
  IdleTime=0;
  count=0;
  MinProc=id;
  MinLoad=0;
  Mindex=0;
  prevovt=0;
  created=0;
  processed=0;
  Inactivity_Detected=0;
  neighborlist=NULL;
  loadlist=NULL;  
  maxNeighbors=0;
  if (!(id)) maxObj=maxObjects;  
  findneighbors();
 
  // Send two messages to objects 
  if(!(id)){

  for (i=1; i<=2; i++) {
    computeMsg *m1 = new computeMsg;
    m1->data=data-i;
    m1->type=WORK_MSG;
    m1->sum=0;
    if(m1->data>-1){
    	ldminavg(avg);
	if (load<avg) recvid=id;
        else recvid=MinProc;
        if (DEBUG) printf("I m in constructor now\n"); 
	created++;
        if (DEBUG) printf("Creating on %d , created no. %d, value: %d\n", id,created,m1->data);
        POSE_invoke(receiveWork(m1), schedulerObject, recvid,0);
	}
    else delete(m1);
   }
 
   computeMsg *m2=new computeMsg;
   m2->data=created;
   m2->sum=processed;
   m2->type=2;
   POSE_invoke(sendData(m2), schedulerObject, (id+1)%maxObjects, 0);
  }
  //to begin work on each scheduler poser
  computeMsg *m2=new computeMsg;
  m2->data=0;
  m2->sum=0;
  m2->type=1;
  POSE_local_invoke(startWork(m2),0);

  //periodic load balancing 
  computeMsg *m1 = new computeMsg;
  m1->type = LDBAL_MSG;
  m1->data=0;
  m1->sum=id;
  POSE_local_invoke(receiveWork(m1), 2);    
}

void schedulerObject::fibonacci(long int key)
{
   int i,avg=0,recvid;
   if (DEBUG) printf("Calculating fibonacci of %ld\n",key); 
   if (key>1){
         for (i=1; i<=2; i++)
         {
            computeMsg *m1 = new computeMsg;
            m1->data=key-i;
            m1->type=WORK_MSG;
            if(m1->data>-1){
                ldminavg(avg);
		if (load<avg) recvid=id;
                else recvid=MinProc;
 		created++;
                if (DEBUG) printf("Creating on %d , created no. %d, value: %d\n", id,created,m1->data);
                POSE_invoke(receiveWork(m1), schedulerObject, recvid,0);
	    }
            else delete(m1);
         }
   }   
   //printf("In Fibonacci Id: %d Created: %d Processed: %d\n",id,created,processed);                                                            
}

void schedulerObject::receiveWork(computeMsg *m) {
  int i,workDone;
  long int key;
  // appfn=&fibonacci;
  srand48(36);
  if (DEBUG) printf("Recieve Work on poser %d at ovt %d\n",id,ovt); 
  if (m->type==LDBAL_MSG) {
     for (i=0; i<maxNeighbors; i++) {
      //reusing fields of computeMsg for data transfer..define new messagetype later.
      computeMsg *m1= new computeMsg;
      m1->type=load;
      m1->data=i;
      m1->sum=id;
      POSE_invoke(recvLoad(m1), schedulerObject, neighborlist[i], maxNeighbors);
    }
   }
  else
  if(m->data>-1){
    key=m->data;
    myenqueue(queue,key);
    load=myqlength(queue);
  }
}

void schedulerObject::startWork(computeMsg *m)
{
  int msum,mtype,mtimestamp,workDone,idle=1;
  long int key=-1;
  msum=m->sum;
  mtype=m->type;
  if (DEBUG) printf("Startwork on object %d at ovt %d\n",id,ovt); 
  if(!(myisempty(queue)))
  {
         key=mydequeue(queue);
         load=myqlength(queue);
         mtimestamp=m->timestamp;
         idle=0;
	 if(key>=0){
		 //(*appfn)(key);
                 processed++;
                 if (DEBUG) printf("Processing on %d , processed no. %d, value: %d\n", id,processed,key);
	         fibonacci(key);
	     	 if (key<=1)
	     	    workDone = 25;
		 else
		         workDone = 15;
		 elapse(workDone);
                 parent->CommitPrintf("%d %d %d\n",id,ovt-workDone,ovt);

		 WorkTime+=workDone;
	 }
	 if (ovt>lastovt+100){
	      float busy;
	      //busy=float(WorkTime*100)/(WorkTime+IdleTime);	    
              //if (ovt-lastovt>200)
	      //{
		//ovt
	      //}
              //{
	      //busy=float(WorkTime*100)/(ovt-lastovt);
               //parent->CommitPrintf("%d %d %d %d %f\n",id,lastovt,ovt-ovt%100,WorkTime,WorkTime*((double)(ovt-ovt%100-lastovt)/(ovt-lastovt)));
               //parent->CommitPrintf("%d %d %d %f\n",id,ovt-ovt%100,ovt, WorkTime-WorkTime*((double)(ovt-ovt%100-lastovt)/(ovt-lastovt)));
                 
                 //parent->CommitPrintf("%d %d %lf\n",id,lastovt/100+1, WorkTime*((double)(ovt-ovt%100-lastovt)/(ovt-lastovt)));
                // parent->CommitPrintf("%d %d %lf\n",id,(ovt-ovt%100)/100+1, WorkTime-WorkTime*((double)(ovt-ovt%100-lastovt)/(ovt-lastovt)));

              //}
	      lastovt=ovt;
	      WorkTime=0;
	      IdleTime=0;	     
	  }
   }
   if (!Inactivity_Detected)
	{                                                       
	   computeMsg *m1 = new computeMsg;
	   m1->data = 0;
	   m1->sum = 0;
	   m1->type = WORK_MSG;
	   IdleTime+=idle*5;
	   POSE_local_invoke(startWork(m1),idle*5);
	   if (ovt>prevovt+50){
	      computeMsg *m1 = new computeMsg;
	      m1->type = LDBAL_MSG;
	      POSE_local_invoke(receiveWork(m1), 50);    
	   }
	   prevovt=ovt;
	}
   else 
         CkExit();
	
}

void schedulerObject::recvLoad(computeMsg *m)
{   
   int mdata,mtype;
   mdata=m->data;
   mtype=m->type;
   loadlist[mdata]=mtype;
   IdleTime+=maxNeighbors;
   ldbalance();
}

void schedulerObject::ldminavg(int& k)
{
  int sum=0, i;
  int mype=id;
  static int start=-1;
  if (start == -1)
    start = CmiMyPe() % (maxNeighbors);
  MinProc = neighborlist[start];
  MinLoad = loadlist[start];
  sum =loadlist[start];
  Mindex = start;
  for (i=1; i<maxNeighbors; i++) {
    start = (start+1) % maxNeighbors;
    sum += loadlist[start];
    if (MinLoad >loadlist[start]) {
      MinLoad = loadlist[start];
      MinProc = neighborlist[start];
      Mindex = start;
    }
  }
  start = (start+2) % maxNeighbors;
  sum += CldLoad();
  if (CldLoad() < MinLoad) {
    MinLoad = CldLoad();
    MinProc = CmiMyPe();
  }
  k = (int)(1.0 + (((float)sum) /((float)(maxNeighbors)+1)));  
}

void schedulerObject::ldbalance()
{
  int sum=0, i, j, overload, numToMove=0, avgLoad=0;
  int totalUnderAvg=0, numUnderAvg=0, maxUnderAvg=0;
  int mype=id;
  count=0;
  ldminavg(avgLoad);
  overload = CldLoad() - avgLoad;
  if (overload > MAXOVERLOAD) {
    for (i=0; i<maxNeighbors; i++)
      if (loadlist[i] < avgLoad) {
        totalUnderAvg += avgLoad-loadlist[i];
        if (avgLoad - loadlist[i] > maxUnderAvg)
          maxUnderAvg = avgLoad - loadlist[i];
        numUnderAvg++;
      }
    if (numUnderAvg > 0)
      for (i=0; ((i<maxNeighbors) && (overload>0)); i++) {
      	j = (i+Mindex)%maxNeighbors;
        if (loadlist[j] < avgLoad) {
          numToMove = avgLoad - loadlist[j];
          if (numToMove > overload)
            numToMove = overload;
          overload -= numToMove;
          loadlist[j]+=numToMove;
          long int key;
          for (i=0;i<numToMove;i++){
	   if(!(myisempty(queue))){
  		key=mydequeue(queue);  
   		computeMsg *m1 = new computeMsg;
   		m1->data=key;
   		m1->type=WORK_MSG;
   		POSE_invoke(receiveWork(m1), schedulerObject, (neighborlist[j]), 0);
   	       }
 	   }
          
        }
      }
  }
}


//extern "C" void gengraph(int, int, int, int *, int *, int, int);

void schedulerObject::findneighbors()
{
  LBTopology *topo;
  LBtopoFn topofn = LBTopoLookup(lbtopo);
  //printf("lbtopo:%s\n",lbtopo);
  if (topofn == NULL) {
    if (id==0) printf("LB> Fatal error: Unknown topology: %s.\n", lbtopo);
    CkExit();
  }
  topo = topofn();
  maxNeighbors = topo->max_neighbors();
  neighborlist = new int[maxNeighbors];
  int nb=0;
  topo->neighbors(id, neighborlist, nb);
  maxNeighbors=nb;
  loadlist = new int[nb];
  for(int i=0;i<nb;i++)  loadlist[i]=0;
}

void schedulerObject::sendData(computeMsg *m)
{
        if (!(id))
		{
			 if (DEBUG) printf("\nId: %d Created: %d Processed: %d\n",id,m->data,m->sum); 
			 if (m->data==m->sum) Inactivity_Detected=1;
			 else   {
				 computeMsg *m1 = new computeMsg;
	  	 		 m1->data = created;
				 m1->sum = processed;
	        		 POSE_invoke(sendData(m1), schedulerObject, (id+1)%maxObjects, 0);
				}
		}
	else 	
		{
			computeMsg *m1 = new computeMsg;
	  	 	m1->data = m->data+created;
			m1->sum = m->sum+processed;
	        	POSE_invoke(sendData(m1), schedulerObject, (id+1)%maxObjects, 0);
	   	}
}
