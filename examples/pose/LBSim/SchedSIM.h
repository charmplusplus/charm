#ifndef _SCHEDSIM_H_
#define _SCHEDSIM_H_

#include <math.h>
#include "queueing.h"
#include "typedefs.h"

#define WORK_MSG 1
#define LDBAL_MSG 2
#define MAXOVERLOAD 2

typedef void (*AppFn)(int n);

// messages
class SchedulerData { // scheduler object constructor message
 public:
  int maxObjects;
  int connectivity;
  char * lbtopo;
  int lbtopolen;
  int id;
  long int data;
  int sum;
  //SchedulerData(){
  //lbtopo=NULL;
  //}
  SchedulerData& operator=(const SchedulerData& obj) {
    eventMsg::operator=(obj);
    maxObjects = obj.maxObjects;
    connectivity = obj.connectivity;
    lbtopolen = obj.lbtopolen;
    for (int i=0;i<lbtopolen;i++) lbtopo[i]=obj.lbtopo[i]; //can use strcpy too?
    id = obj.id;
    data = obj.data;
    sum = obj.sum;
    return *this;
  }
};

class computeMsg {
 public:
  long int data;
  int sum;
  int type;
  computeMsg& operator=(const computeMsg& obj) {
    eventMsg::operator=(obj);
    data = obj.data;
    sum = obj.sum;
    type = obj.type;
    return *this;
  }
};

// posers
class schedulerObject {
  int maxObjects;
  int connectivity;
  int id;
  long int data;
  int count;
  //For stats collection
  int WorkTime;
  int IdleTime;
  int lastovt;
  //FILE *fp;
  //Neighborlist 
  int maxNeighbors;
  int *neighborlist;
  int *loadlist;
  //For load balancing
  int MinProc;
  int MinLoad;
  int Mindex; 
  int prevovt;
  //Queue data structure
  QUEUE *queue;
  //Application function
  AppFn appfn;
  //Load (Queue Length)
  int load;
  //LoadBalancing Topology
  char * lbtopo;
  int lbtopolen;
  //To detect termination
  int created;
  int processed;
  int Inactivity_Detected;
public:
  // The Essentials
  schedulerObject() {}
  schedulerObject(SchedulerData *m);
  ~schedulerObject() 
  {
    free(neighborlist);
    free(loadlist);
    free(lbtopo);
    //free(fp);
    while(!(myisempty(queue))){
                  mydequeue(queue);
    }
    free(queue->front);
    delete(queue);
  }
  schedulerObject& operator=(const schedulerObject& obj) {
    chpt<state_schedulerObject>::operator=(obj);
    maxObjects = obj.maxObjects;
    connectivity = obj.connectivity;
    id = obj.id;
    data = obj.data;
    count = obj.count;    
    maxNeighbors=obj.maxNeighbors;
    load=obj.load;
    WorkTime=obj.WorkTime;
    IdleTime=obj.IdleTime;
    lastovt=obj.lastovt;
    MinProc=obj.MinProc;
    MinLoad=obj.MinLoad;
    Mindex=obj.Mindex;
    prevovt=obj.prevovt;
    created=obj.created;
    processed=obj.processed;
    lbtopolen=obj.lbtopolen;
    Inactivity_Detected=obj.Inactivity_Detected;
    /*fp=(FILE *)malloc(sizeof(FILE));
    fp=obj.fp;*/
    neighborlist=(int *)malloc(sizeof(int)*maxNeighbors);
    loadlist=(int *)malloc(sizeof(int)*maxNeighbors);
    for(int i=0;i<maxNeighbors;i++){
		 neighborlist[i]=obj.neighborlist[i];
		 loadlist[i]=obj.loadlist[i];
		}
    lbtopo=(char*)malloc(sizeof(char)*lbtopolen);
    NODE* ptr;
    ptr=obj.queue->front;
    queue= new QUEUE;
    myinitialise(queue);
    while (ptr!=obj.queue->rear) {
	          ptr=ptr->next;
               	  myenqueue(queue,ptr->data);
		} 
    //printf("I am here"); 	
    return *this;
  }
  void pup(PUP::er &p) {
    chpt<state_schedulerObject>::pup(p);
    p(maxObjects);
    p(connectivity);
    p(id);
    p(data);
    p(count);
    p(maxNeighbors);
    p(lbtopolen);
    p(load);
    p(WorkTime);
    p(IdleTime);
    p(lastovt);
    p(MinProc);
    p(MinLoad);
    p(Mindex);
    p(prevovt);
    p(created);
    p(processed);
    p(Inactivity_Detected);
    //p(fp);
    if (p.isUnpacking()) {
		neighborlist=(int *)malloc(sizeof(int)*maxNeighbors);
                loadlist=(int *)malloc(sizeof(int)*maxNeighbors);
                lbtopo=(char*)malloc(sizeof(char)*lbtopolen);
		//fp=(FILE *)malloc(sizeof(FILE));
                //queue=(NODE *)malloc(sizeof(NODE)*load);
                }
  }
  // Event methods
  void receiveWork(computeMsg *m);
  void receiveWork_anti(computeMsg *) { restore(this); }
  void receiveWork_commit(computeMsg *) { /*printf("Commiting event receiveWork for %d now\n",id);*/}
  
  void startWork(computeMsg *m);
  void startWork_anti(computeMsg *) { restore(this); }
  void startWork_commit(computeMsg *) { /*printf("Commiting event startWork for %d now\n",id);*/}

  void recvLoad(computeMsg *m);
  void recvLoad_anti(computeMsg *) { restore(this); }
  void recvLoad_commit(computeMsg *){ /*printf("Commiting event recvLoad for %d now\n",id);*/}

  void sendData(computeMsg *m);
  void sendData_anti(computeMsg *) { restore(this); }
  void sendData_commit(computeMsg *){ /*printf("Commiting event sendData for %d now\n",id);*/} 

  //local functions
  void ldbalance();
  void ldminavg(int&);
  void fibonacci(long int);
  void findneighbors();
 };

#endif

