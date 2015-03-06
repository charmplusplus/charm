/** \file RecBipartLB.C
 *  Author: Swapnil Ghike
 *  Date Created: 
 *  E-mail: ghike2@illinois.edu
 *
 *  This strategy does a recursive bipartition of the object graph and the
 *  processor graph. The objects are divided based on the loads in case of odd
 *  number of processors. Recursive bi-partitioning is done by a breadth-first
 *  traversal until you have the required load in one group.
 *
 * 
 *  At each recursive biparititioning, the boundaries are refined to minimize
 *  the edge cut. Vertices are moved across the boundary to see if the edge cut
 *  reduces while trying to maintain proportionate load in both partitions.
 */

/**
 *  \addtogroup CkLdb
 */

/*@{*/

#include "RecBipartLB.h"
#include "ckgraph.h"
#include <limits>
#include <queue>
#include <vector>

using std::vector;

/**
 *  Class to contain additional data about the vertices in object graph
 */
class Vertex_helper {
  public:
    inline int getPartition(){ return partition; }
    inline void setPartition(int p){partition=p; }
    inline bool getMarked(){ return marked; }
    inline void setMarked(bool v){ marked=v;}
    inline bool getBoundaryline(){return boundaryline;}
    inline void setBoundaryline(bool v){ boundaryline=v;}
    inline int getEdgestopart1(){return edgestopart1;}
    inline int getEdgestopart2(){return edgestopart2;}
    inline void setEdgestopart1(int v){edgestopart1=v;}
    inline void setEdgestopart2(int v){edgestopart2=v;}
    inline void incEdgestopart1(int v){edgestopart1+=v ;}
    inline void incEdgestopart2(int v){edgestopart2+=v;}
    inline void decEdgestopart1(int v){edgestopart1-=v;}
    inline void decEdgestopart2(int v){edgestopart2-=v;}
    inline void setLevel(int l){level=l;}
    inline int getLevel(){return level;}
    inline int getGain(){return gain;}
    inline void setGain(int v){gain=v;};

  private:
    int partition;      // partition to which this vertex currently belongs
    bool marked;       // already marked or not
    bool boundaryline;  //on boundaryline of a partition or not
    int edgestopart1; //only for boundaryline vertices
    int edgestopart2; //only for boundaryline vertices
    int gain;		//gain if this vertex switched partitions
    int level;
};

/**
 *  Class to handle the boundaries of child partitions
 */
class BQueue {
  public:
    vector<int> q;

    BQueue(short b){
      forboundary=b;
    }

    inline int getMingain(){return mingain;}
    inline void setMingain(int v){mingain=v;}
    inline int getVertextoswap(){return vertextoswap;}
    inline void setVertextoswap(int v){vertextoswap=v;}
    inline int getSwapid(){return swapid;}
    inline void setSwapid(int v){swapid=v;}
    inline short getBoundary(){return forboundary;}
    void push(Vertex *);
    void removeComplete(Vertex *);
    void removeToSwap(Vertex *);

  private:
    int mingain;
    int vertextoswap;
    int swapid;
    short forboundary;
};

void RecursiveBiPart(ObjGraph *, vector<Vertex *> & ,int, int);
void adjustqueues(ObjGraph *, BQueue *, BQueue *, vector<Vertex *> &, vector<Vertex *> &,int *, int);
void adjustgain(ObjGraph *,vector<Vertex *> &,BQueue *);
void RefineBoundary(ObjGraph *, vector<Vertex *> &, vector<Vertex *> &,BQueue *, BQueue *, int, int, int, double, double, double);
int modifypartitions(ObjGraph *,vector<Vertex *> &, vector<Vertex *> &, BQueue *, BQueue *, int, int);
void swapQ1toQ2(ObjGraph *, BQueue *, BQueue *, int);
Vertex * removeinSwap(ObjGraph *,BQueue *, BQueue *, int);
Vertex * removePtr(vector<Vertex *> &,int);

int level;
double TOTALLOAD;
vector<Vertex_helper *> vhelpers;
int numparts, peno;
ProcArray *parray;

CreateLBFunc_Def(RecBipartLB, "Algorithm for load balacing based on recursive bipartitioning of object graph")

//removes from BQueue but not from boundaryline
void BQueue::removeToSwap(Vertex *vert)
{
  int i;
  int v=vert->getVertexId();
  for(i=0;i<q.size();i++)
  {
    if(q[i]==v)
    {
      //boundaryline and edgestopart1, edgestopart2 are left as they were since this vertex only swaps boundarylines
      q[i]=q.back();
      q.pop_back();
      return;
    }
  }
}

//completely removes from the BQueue as well as from both boundarylines
void BQueue::removeComplete(Vertex *vert)
{	 
  int i;
  int v=vert->getVertexId();
  for(i=0;i<q.size();i++)
  {
    if(q[i]==v)
    {
      vhelpers[v]->setBoundaryline(false);
      vhelpers[v]->setEdgestopart1(0);
      vhelpers[v]->setEdgestopart2(0);
      q[i]=q.back();
      q.pop_back();
      return;
    }
  }
}

void BQueue::push(Vertex *vert)
{
  int v=vert->getVertexId();
  q.push_back(v);
  vhelpers[v]->setBoundaryline(true);
}


RecBipartLB::RecBipartLB(const CkLBOptions &opt) : CBase_RecBipartLB(opt) {
  lbname = "RecBipartLB";
  if(CkMyPe() == 0)
    CkPrintf("RecBipartLB created\n");
}

bool RecBipartLB::QueryBalanceNow(int _step) {
  return true;
}

void RecBipartLB::work(LDStats *stats) {
  vector<Vertex *> ptrvector;
  /** ========================== INITIALIZATION ============================= */
  ProcArray *parr = new ProcArray(stats);	// Processor Array
  ObjGraph *ogr = new ObjGraph(stats);		// Object Graph


  /** ============================= STRATEGY ================================ */
  level=0;
  peno=0;
  TOTALLOAD=0;
  numparts=CkNumPes();
  parray=parr;

  double avgLoad = parr->getAverageLoad();
  int numPes = parr->procs.size();

  parr->resetTotalLoad();
  for(int i=0;i<ogr->vertices.size();i++)
  {
    Vertex_helper *helper = new Vertex_helper();
    vhelpers.push_back(helper);
    ptrvector.push_back((Vertex *)&(ogr->vertices[i]));

  }

  RecursiveBiPart(ogr,ptrvector,1,numparts);

  /** ============================== CLEANUP ================================ */
  ogr->convertDecisions(stats);		// Send decisions back to LDStats
}

/* Function that performs Recursive bipartitioning of the object graph.*/
void RecursiveBiPart(ObjGraph *ogr, vector<Vertex *> &pvertices, int parent, int nump)
{
  //if the number of processors that this call has to deal with is 1, dont recurse any further
  if(nump==1)
  {
    parray->procs[peno].totalLoad() = 0.0;
    for(int i=0;i<pvertices.size();i++)
    {
      pvertices[i]->setNewPe(peno);
      parray->procs[peno].totalLoad() += pvertices[i]->getVertexLoad();
    }
    peno++;

    return;
  }

  int numerator=nump/2;
  double ratio = ((double)numerator/nump);//(ratio=floor of nump/2 divided by nump) This is equal to half if nump is even

  //if you have only one vertex in the parent partition, just map it to the appropriate processor
  if(pvertices.size()==1)
  {
    level++;	
    RecursiveBiPart(ogr,pvertices,2*parent-1,numerator);//nump =6 =>numerator =3, nump =7=>numertaor =3
    level--;
    return;
  }

  //child partitions
  vector<Vertex *> partition1;
  vector<Vertex *> partition2;
  vector<bool> taken(vhelpers.size(), false);

  int start = pvertices[0]->getVertexId();
  int count=0;
  double loadseen=0;
  double pload=0;
  bool getout=false;
  std::queue<int> que2;
  std::queue<int> que1;
  int KLFMruns=pvertices.size()/5;

  //initialize from the parent partition
  for(int i=0;i<pvertices.size();i++){
    int id=pvertices[i]->getVertexId();
    vhelpers[id]->setPartition(2*parent);
    vhelpers[id]->setMarked(false);
    vhelpers[id]->setBoundaryline(false);
    vhelpers[id]->setEdgestopart2(0);
    vhelpers[id]->setEdgestopart1(0);
    vhelpers[id]->setLevel(level);
    pload+=ogr->vertices[id].getVertexLoad();
  }

  // start at vertex with id 0
  que2.push(start);
  vhelpers[start]->setMarked(true);

  int i, nbr, lastforced=0;
  int visitcount=0;

  bool swap=true;
  int ei=-1;
  // breadth first traversal
  while(!que2.empty() && !getout) {
    int n = que2.front();
    que2.pop();
    count++;

    Vertex *v=(Vertex *)&(ogr->vertices[n]);
    loadseen+=v->getVertexLoad();

    vhelpers[v->getVertexId()]->setPartition(2*parent-1);//vertices in que2 are in the other partition

    partition1.push_back(v);
    taken[v->getVertexId()]=true;

    //this case is useful if the last remaining vertex is way too large/heavy
    if(count==pvertices.size()-1)
      break;

    //visit neighbors of a vertex
    while(1) {
      ei++;
      if(swap==true && ei==v->sendToList.size())
      { swap=false; ei=0;}

      if(swap==false && ei==v->recvFromList.size())
      {swap=true; ei=-1;  break;}

      if(swap)
	nbr = v->sendToList[ei].getNeighborId();
      else
	nbr=v->recvFromList[ei].getNeighborId();

      Vertex_helper *u=(vhelpers[nbr]);
      visitcount++;

      //not all neighbos of v belong to the parent partition
      if((u->getMarked()==false) && (u->getPartition()==2*parent) && (u->getLevel()==level)){
	que2.push(nbr);
	u->setMarked(true);

      }//end of if
    }//end of while(1)loop

    //if you have visited enough vertices, stop Breadth First traversal
    if(loadseen>=(ratio*pload))//if nump is even, ratio=1/2, if nump is odd say 7, ratio = 3/7. 1st rec call will have nump =3 and second nump = 4
    {
      getout=true;
    }
    else{
      //if the parent partition is disconnected (likely to happen down the recursion tree), force a vertex in BFS
      if(que2.empty())
      {
	for(int i=lastforced;i<pvertices.size();i++)
	{
	  Vertex *w=pvertices[i];
	  if(taken[w->getVertexId()]==false)
	  {
	    que2.push(w->getVertexId());
	    vhelpers[w->getVertexId()]->setMarked(true);
	    lastforced=i+1;
	    break;
	  }		
	}
      }
    }
  } // end of while loop


  for(int i=0;i<pvertices.size();i++){
    Vertex *v=pvertices[i];
    if(taken[v->getVertexId()]==false)
      partition2.push_back(v);
  }

  int initialedgecut;

  //Boundaries in respective child partitions, they are really vectors though the name says BQueue
  BQueue *q1=new BQueue(1);
  BQueue *q2=new BQueue(2);
  int tempsize=que2.size();

  for(i=0;i<tempsize;i++)
  {
    q2->push((Vertex *)&(ogr->vertices[que2.front()]));//also sets boundaryline=true for each vertex
    que2.pop();
  }
  adjustqueues(ogr,q1,q2,partition1, partition2, &initialedgecut,parent);//adjusts initial queues and gains, edgecut

  RefineBoundary(ogr,partition1,partition2,q1,q2,KLFMruns,initialedgecut,parent,loadseen,pload-loadseen,ratio);//iteratively modified queues and gains, edgecuts

  //level must be incremented/decremented here
  level++;
  RecursiveBiPart(ogr,partition1,vhelpers[partition1[0]->getVertexId()]->getPartition(),numerator);//nump =6 =>numerator =3, nump =7=>numertaor =3
  RecursiveBiPart(ogr,partition2,vhelpers[partition2[0]->getVertexId()]->getPartition(), nump-numerator);//nump=6=>nump-numerator=3, nump=7=>nump-numerator=4
  level--;
}

//Fills in que1, que2 and adjusts their gains, calculates initial edgecut before KLFM
void adjustqueues(ObjGraph *ogr, BQueue *que1, BQueue *que2, vector <Vertex *> &partition1, vector<Vertex *> &partition2, int *initialedgecut, int parent)
{
  int i,uid,wid;
  bool swap=true;
  int ei=-1;
  Edge *edge;
  int edgecut=0;
  que2->setMingain(std::numeric_limits<int>::max());
  que2->setVertextoswap(-1);
  que2->setSwapid(-1);

  //This loop fills in que1 and adjusts gain of que2
  for(i=0;i<que2->q.size();i++)//for each vertex v in que2
  {
    int vid=que2->q[i];
    Vertex *v=((Vertex *)&(ogr->vertices[vid]));

    while(1) {
      ei++;
      if(swap==true && ei==v->sendToList.size())
      { swap=false; ei=0;}

      if(swap==false && ei==v->recvFromList.size())
      { swap=true; ei=-1; break;}

      if(swap)
      {
	uid = v->sendToList[ei].getNeighborId();
	edge=(Edge *)&(v->sendToList[ei]);
      }
      else
      {
	uid = v->recvFromList[ei].getNeighborId();
	edge = (Edge *)&(v->recvFromList[ei]);
      }

      Vertex *u =(Vertex *)&(ogr->vertices[uid]);

      if((vhelpers[uid]->getPartition())==(2*parent-1) && (vhelpers[uid]->getLevel())==level)//since v is on boundaryline2, its every neighbour in part1 is on boundaryline1
      {
	//if not already added to que1
	if(vhelpers[uid]->getBoundaryline()==false)
	{
	  que1->push(u);//also sets boundaryline=true
	}

	//calculate edgecut
	edgecut+=edge->getNumBytes();
	vhelpers[vid]->incEdgestopart1(edge->getNumBytes());//assuming it was initialized earlier to 0
	vhelpers[uid]->incEdgestopart2(edge->getNumBytes());//assuming it was initialized earlier to 0
      }
      if(vhelpers[uid]->getPartition()==2*parent && vhelpers[uid]->getLevel()==level)
      {
	vhelpers[vid]->incEdgestopart2(edge->getNumBytes());
      }
    }//end of while(1) loop


    //Edge counts are initialized while performing BFS
    vhelpers[vid]->setGain(vhelpers[vid]->getEdgestopart2() - vhelpers[vid]->getEdgestopart1());
    if(vhelpers[vid]->getGain() < que2->getMingain())//we want most negative gain
    {
      que2->setMingain(vhelpers[vid]->getGain());
      que2->setVertextoswap(v->getVertexId());
      que2->setSwapid(i);
    }
  }

  for(i=0;i<que1->q.size();i++)
  {
    int uid=que1->q[i];
    swap = true; ei=-1;
    Vertex *u=(Vertex *)&(ogr->vertices[uid]);

    while(1) {
      ei++;
      if(swap==true && ei==u->sendToList.size())
      { swap=false; ei=0;}

      if(swap==false && ei==u->recvFromList.size())
      { swap=true; ei=-1; break;}

      if(swap)
      {
	wid=u->sendToList[ei].getNeighborId();
	edge = (Edge *)&(u->sendToList[ei]);
      }
      else
      {
	wid=u->recvFromList[ei].getNeighborId();
	edge = (Edge *)&(u->recvFromList[ei]);
      }

      Vertex *w=(Vertex *)&(ogr->vertices[wid]);
      if(vhelpers[wid]->getLevel()==level && vhelpers[wid]->getPartition()==(2*parent-1))
	vhelpers[uid]->incEdgestopart1(edge->getNumBytes());
    }

  }	
  *initialedgecut=edgecut;
  //figure out which vertex to swap out of boundaryline1
  //by this time we know edgestopart2 for every vertex in que1
  adjustgain(ogr,partition1, que1);
}

//precondition - edgestopart1 and edgestopart2 must be known for every vertex in queue
void adjustgain(ObjGraph *ogr,vector<Vertex *> &partition, BQueue *que)
{
  int i;
  int bdry=que->getBoundary();
  que->setMingain(std::numeric_limits<int>::max());
  que->setVertextoswap(-1);
  que->setSwapid(-1);

  for (i=0;i<que->q.size();i++)//for each vertex u in que
  {
    int uid=que->q[i];
    Vertex *u =(Vertex *)&(ogr->vertices[uid]);

    if(bdry==1)
    {
      vhelpers[uid]->setGain(vhelpers[uid]->getEdgestopart1() - vhelpers[uid]->getEdgestopart2());
    }
    else if(bdry==2)
    {
      vhelpers[uid]->setGain(vhelpers[uid]->getEdgestopart2() - vhelpers[uid]->getEdgestopart1());
    }
    if(vhelpers[uid]->getGain() < que->getMingain())//we want most negative gain
    {	
      que->setMingain(vhelpers[uid]->getGain());
      que->setVertextoswap(u->getVertexId());
      que->setSwapid(i);
    }
  }
}

// Fiduccia Mattheyses boundary refinement algorithm
void RefineBoundary(ObjGraph *ogr,vector<Vertex *> &partition1, vector<Vertex *> &partition2, BQueue *que1, BQueue *que2, int runs, int initialedgecut, int parent, double part1load, double part2load, double ratio)
{
  int r=runs;
  int t;
  if(que1->q.size()<que2->q.size())
    t=que1->q.size();
  else t=que2->q.size();
  if(t<r) r=t;
  if(r==0) return;

  int newedgecut=initialedgecut;

  for(int i=0;i<r;i++)
  {
    if((part1load/(part1load+part2load))>ratio)
    {
      if(partition1.size()>1 && que1->q.size()>0)//because if part1 has only one vertex which is heavier than the whole part2, swapping it wouldnt make sense
      {
	double xfer=(ogr->vertices[que1->getVertextoswap()]).getVertexLoad();
	part1load-=xfer;
	part2load+=xfer;
	newedgecut=modifypartitions(ogr, partition1, partition2, que1, que2, newedgecut, parent);//it also should adjust the new gains of both boundaries and the edgecut
      }
    }
    else{
      if(partition2.size()>1 && que2->q.size()>0)
      {
	double xfer=(ogr->vertices[que2->getVertextoswap()]).getVertexLoad();
	part2load-=xfer;
	part1load+=xfer;
	newedgecut=modifypartitions(ogr, partition1, partition2, que2, que1, newedgecut, parent);//it also should adjust the new gains of both boundaries and the edgecut
      }
    }

  }
}


int modifypartitions(ObjGraph *ogr,vector<Vertex *> &partition1, vector<Vertex *> &partition2, BQueue *q1, BQueue *q2,int ec, int parent)
{
  int newedgecut;
  if(q1->getBoundary()==1)//we are swapping vertex out of boundaryline1
  {
    int e2=vhelpers[q1->getVertextoswap()]->getEdgestopart2();
    int e1=vhelpers[q1->getVertextoswap()]->getEdgestopart1();
    newedgecut=ec-(e2)+(e1);
    vhelpers[q1->getVertextoswap()]->setPartition(2*parent);
    Vertex *ptr=removePtr(partition1,q1->getVertextoswap());
    partition2.push_back(ptr);

  }
  else if(q1->getBoundary()==2)//we are swapping vertex out of boundaryline2
  {
    int e1=vhelpers[q1->getVertextoswap()]->getEdgestopart1();
    int e2=vhelpers[q1->getVertextoswap()]->getEdgestopart2();
    newedgecut=ec-(e1)+(e2);
    vhelpers[q1->getVertextoswap()]->setPartition(2*parent-1);
    Vertex *ptr=removePtr(partition2,q1->getVertextoswap());
    partition1.push_back(ptr);

  }

  swapQ1toQ2(ogr,q1,q2,parent);//avoid thrashing, same vertex cannot be swapped more than once in same call
  adjustgain(ogr,partition1,q1);//not required for last run of KLFM
  adjustgain(ogr,partition2,q2);	//not required for last run of KLFM
  return newedgecut;
}

void swapQ1toQ2(ObjGraph *ogr, BQueue *q1, BQueue *q2, int parent)
{
  Vertex *vert=removeinSwap(ogr,q1,q2,parent);//remove vertex from q1
  //removevert also removes or brings in new vertices in the queues, so the edgestopart1 and edgestopart2 are calculated for new vertices inside removevert
  q2->push(vert);
}

Vertex * removeinSwap(ObjGraph *ogr,BQueue *q1, BQueue *q2, int parent)
{
  int ei=-1, uid, wid, einested=-1;
  Edge *edge, *edgenested;
  bool swap=true, swapnested=true;
  Vertex *v=(Vertex *)&(ogr->vertices[q1->getVertextoswap()]);
  //edge counts of v do not change
  //Adjust edgecounts of neighbours, verify whether any additions or deletions happen to the boundarylines

  while(1) {
    ei++;
    if(swap==true && ei==v->sendToList.size())
    { swap=false; ei=0;}
    if(swap==false && ei==v->recvFromList.size())
    { swap=true; ei=-1; break;}
    if(swap)
    {
      uid=v->sendToList[ei].getNeighborId();
      edge=(Edge *)&(v->sendToList[ei]);
    }
    else
    {
      uid=v->recvFromList[ei].getNeighborId();
      edge=(Edge *)&(v->recvFromList[ei]);
    }

    Vertex *u=(Vertex *)&(ogr->vertices[uid]);

    if(q1->getBoundary()==1)//vertex being removed out of boundaryline1
    {
      if((vhelpers[uid]->getBoundaryline()==true) && vhelpers[uid]->getLevel()==level)//do for both partitions
      {
	vhelpers[uid]->incEdgestopart2(edge->getNumBytes());
	vhelpers[uid]->decEdgestopart1(edge->getNumBytes());
      }
      else if(vhelpers[uid]->getPartition()==(2*parent-1) && vhelpers[uid]->getLevel()==level && vhelpers[uid]->getBoundaryline()==false)
	//nbr u of v which was in part1, but not in boundaryline1, is now introduced to boundaryline1
      {
	//new to boundaryline1, hence calculate edgestopart2
	vhelpers[uid]->setEdgestopart2(edge->getNumBytes());
	vhelpers[uid]->setEdgestopart1(0);

	while(1) {
	  einested++;
	  if(swapnested==true && einested==u->sendToList.size())
	  { swapnested=false; einested=0;}
	  if(swapnested==false && einested==u->recvFromList.size())
	  { swapnested=true; einested=-1; break;}
	  if(swapnested)
	  {
	    wid=u->sendToList[einested].getNeighborId();
	    edgenested=(Edge *)&(u->sendToList[einested]);
	  }
	  else
	  {
	    wid=u->recvFromList[einested].getNeighborId();
	    edgenested = (Edge *)&(u->recvFromList[einested]);
	  }
	  Vertex *w=(Vertex *)&(ogr->vertices[wid]);
	  if(vhelpers[wid]->getLevel()==level && vhelpers[wid]->getPartition()==(2*parent-1))
	    vhelpers[uid]->incEdgestopart1(edgenested->getNumBytes());
	}
	q1->push(u);//also sets boundaryline=true
      }
      if(vhelpers[uid]->getPartition()==2*parent && vhelpers[uid]->getLevel()==level && vhelpers[uid]->getBoundaryline()==true && vhelpers[uid]->getEdgestopart1()==0)
	//vertex in part2, on boundaryline2, now not a part of boundaryline2
      {
	q2->removeComplete(u);//q1 is queue1, q2 is queue2//sets boundaryline=false
      }
    }
    else if(q1->getBoundary()==2)//vertex being removed out of boundaryline2
    {
      if(vhelpers[uid]->getBoundaryline()==true && vhelpers[uid]->getLevel()==level)//do for both partitions
      {
	vhelpers[uid]->incEdgestopart1(edge->getNumBytes());
	vhelpers[uid]->decEdgestopart2(edge->getNumBytes());
      }
      else if(vhelpers[uid]->getPartition()==2*parent && vhelpers[uid]->getLevel()==level && vhelpers[uid]->getBoundaryline()==false)
	//vertex which was in part2, but not in boundaryline2, is now introduced to boundaryline2
      {
	//new to boundaryline2
	vhelpers[uid]->setEdgestopart1(edge->getNumBytes());
	vhelpers[uid]->setEdgestopart2(0);

	while(1) {
	  einested++;
	  if(swapnested==true && einested==u->sendToList.size())
	  { swapnested=false; einested=0;}
	  if(swapnested==false && einested==u->recvFromList.size())
	  { swapnested=true; einested=-1; break;}
	  if(swapnested)
	  {
	    wid=u->sendToList[einested].getNeighborId();
	    edgenested = (Edge *)&(u->sendToList[einested]);
	  }
	  else
	  {
	    wid=u->recvFromList[einested].getNeighborId();
	    edgenested= (Edge *)&(u->recvFromList[einested]);
	  }

	  Vertex *w=(Vertex *)&(ogr->vertices[wid]);
	  if(vhelpers[wid]->getLevel()==level && vhelpers[wid]->getPartition()==(2*parent))
	    vhelpers[uid]->incEdgestopart2(edgenested->getNumBytes());
	}

	q1->push(u);//q1 is boundaryline2
      }
      if(vhelpers[uid]->getPartition()==(2*parent-1) && vhelpers[uid]->getLevel()==level && vhelpers[uid]->getBoundaryline()==true && vhelpers[uid]->getEdgestopart2()==0)
	//vertex in part1, on boundaryline1, now not a part of boundaryline1
      {
	q2->removeComplete(u);//q1 is queue1, q2 is queue2
      }
    }
  }

  //remove vertex v from q1 to swap into q2
  /*q1->removeToSwap(v); */
  q1->removeComplete(v);	
  return v;
}

Vertex * removePtr(vector<Vertex *> &vec,int id)
{
  int i;
  for(i=0;i<vec.size();i++)
  {
    if(vec[i]->getVertexId()==id)
    {
      Vertex *ptr=vec[i];
      vec[i]=vec.back();
      vec.pop_back();
      return ptr;
    }
  }
  return NULL;
}

#include "RecBipartLB.def.h"
/*@}*/
