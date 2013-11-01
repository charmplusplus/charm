/**
 * \addtogroup CkLdb
*/
/*@{*/

#include <math.h>

//#include "LBDatabase.h"
#include "cklists.h"
#include "topology.h"

extern "C" char *_lbtopo;			/* topology name string */

int LBTopology::get_hop_count(int src,int dest)
{
	int npe;
	int *visited_srcs;

	if(src==dest)
		return 0;
	
	npe = max_neighbors();
	visited_srcs = new int[npes];
	
 	int count = rec_hop_count(src,dest,npe,1,visited_srcs,999999);
	delete [] visited_srcs;

	return count;
}

int LBTopology::rec_hop_count(int src,int dest,int max_neigh,int count,int *visited_srcs,int min_hop_cnt)
{
	int *pes = new int[max_neigh];
	//int min_hop_cnt=999999;
	int ret_val=0;
	int skip_neigh=0;
	int neigh_cnt=0;
	int i;
	
	neighbors(src,pes,neigh_cnt);
	
	visited_srcs[count-1]=src;
	
	for(i=0;i<neigh_cnt;i++)
	{
		if(pes[i]==dest)
			return count;
	}
	for(i=0;i<neigh_cnt;i++)
	{
		for(int j=0;j<count;j++)
			if(visited_srcs[j]==pes[i])
			{
				skip_neigh=1;
				break;
			}
		if(!skip_neigh)
		{
			if(min_hop_cnt > count+1){
				ret_val=rec_hop_count(pes[i],dest,max_neigh,count+1,visited_srcs,min_hop_cnt);
				if(ret_val < min_hop_cnt)
					min_hop_cnt = ret_val;
			}
		}
		else
			skip_neigh=0;
	}
	delete [] pes;
	return min_hop_cnt;
}

double LBTopology::per_hop_delay(int last_hop)
{
	if(!last_hop)
		return (HOP_LINK_DELAY + HOP_PROC_DELAY);
	else
		return HOP_LINK_DELAY;
}

void LBTopology::get_pairwise_hop_count(double  **distance)
{
  struct queueNode
  {
    int index;
    int dist;
    queueNode *next;
    queueNode(int i,int d): index(i), dist(d), next(NULL) {}
  };
  
  bool *visited=new bool[npes];
  int *neigh=new int[max_neighbors()];
  int num_neighbors;

  for(int i=0;i<npes;i++)
  {
    //Init data structures for BFS from i-th proc
    for(int j=0;j<npes;j++)
      visited[j]=false;

    queueNode *q=new queueNode(i,0);
    queueNode *last=q;
    distance[i][i]=0;
    visited[i]=true;

    // Perform BFS until queue is empty
    while(q)
    { 
      neighbors(q->index,neigh,num_neighbors);
      for(int j=0;j<num_neighbors;j++)
      {
        if(!visited[neigh[j]])
        {
          visited[neigh[j]]=true;
          distance[i][neigh[j]]=q->dist+1;
          queueNode *qnew=new queueNode(neigh[j],q->dist+1);
          last->next=qnew;
          last=last->next;
        }
      }
      queueNode *qtemp=q;
      q=q->next;
      delete qtemp;
    }
  }
  delete[] visited;
  delete[] neigh;
}

//smp - assume 1,2,3 or 4 processors per node

template <int ppn>
class LBTopo_smp_n: public LBTopology {
public:
  LBTopo_smp_n(int p): LBTopology(p) {}
  virtual int max_neighbors() { return npes - 1; }

  virtual void neighbors(int mype, int* _n, int &nb){
      nb = 0;
      for(int i=1; i<=ppn; i++)
      {
          _n[nb++] = (mype+i)%npes;
      }
  }
	
	int get_hop_count(int src,int dest){
		
		//CmiPrintf("in smp get_hop_count\n");
		int a = src/ppn;
		int b = dest/ppn;
		
		if(a!=b){
			//CmiPrintf("2 returned\n");
			return 2;
		}
		else{
			//CmiPrintf("1 returned\n");
			return 1;
		}
	}
};

typedef LBTopo_smp_n<1> LBTopo_smp_n_1;
typedef LBTopo_smp_n<2> LBTopo_smp_n_2;
typedef LBTopo_smp_n<3> LBTopo_smp_n_3;
typedef LBTopo_smp_n<4> LBTopo_smp_n_4;
typedef LBTopo_smp_n<5> LBTopo_smp_n_5;
typedef LBTopo_smp_n<6> LBTopo_smp_n_6;
typedef LBTopo_smp_n<7> LBTopo_smp_n_7;
typedef LBTopo_smp_n<8> LBTopo_smp_n_8;
typedef LBTopo_smp_n<9> LBTopo_smp_n_9;
typedef LBTopo_smp_n<10> LBTopo_smp_n_10;

LBTOPO_MACRO(LBTopo_smp_n_1)
LBTOPO_MACRO(LBTopo_smp_n_2)
LBTOPO_MACRO(LBTopo_smp_n_3)
LBTOPO_MACRO(LBTopo_smp_n_4)
LBTOPO_MACRO(LBTopo_smp_n_5)
LBTOPO_MACRO(LBTopo_smp_n_6)
LBTOPO_MACRO(LBTopo_smp_n_7)
LBTOPO_MACRO(LBTopo_smp_n_8)
LBTOPO_MACRO(LBTopo_smp_n_9)
LBTOPO_MACRO(LBTopo_smp_n_10)


// ring

LBTOPO_MACRO(LBTopo_ring)

int LBTopo_ring::max_neighbors()
{
  if (npes > 2) return 2;
  else return (npes-1);
}

void LBTopo_ring::neighbors(int mype, int* _n, int &nb)
{
  nb = 0;
  if (npes>1) _n[nb++] = (mype + npes -1) % npes;
  if (npes>2) _n[nb++] = (mype + 1) % npes;
}

int LBTopo_ring::get_hop_count(int src,int dest){
	
	int dist=src-dest;
	if(dist<0) dist=-dist;
	
	if((npes-dist) < dist)
		return (npes-dist);
	else
		return dist;
}

//  TORUS 2D

LBTOPO_MACRO(LBTopo_torus2d)

LBTopo_torus2d::LBTopo_torus2d(int p): LBTopology(p) 
{
  width = (int)sqrt(p*1.0);
  if (width * width < npes) width++;
}

int LBTopo_torus2d::max_neighbors()
{
  return 4;
}

int LBTopo_torus2d::goodcoor(int x, int y)
{
  if (x<0 || x>=width) return -1;
  if (y<0 || y>=width) return -1;
  int next = x*width + y;
  if (next<npes && next>=0) return next;
  return -1;
}

static int checkuniq(int *arr, int nb, int val) {
  for (int i=0;i<nb;i++) if (arr[i]==val) return 0;
  return 1;
}

void LBTopo_torus2d::neighbors(int mype, int* _n, int &nb)
{
  int next;
  int x = mype/width;
  int y = mype%width;
  nb=0;
  for (int i=-1; i<=1; i+=2) {
    int x1 = x+i;
    if (x1 == -1) {
      x1 = width-1;
      while (goodcoor(x1, y)==-1) x1--;
    }
    else if (goodcoor(x1, y) == -1) x1=0;
    next = goodcoor(x1, y);
    CmiAssert(next != -1);
    if (next != mype && checkuniq(_n, nb, next)) _n[nb++] = next;

    int y1 = y+i;
    if (y1 == -1) {
      y1 = width-1;
      while (goodcoor(x, y1)==-1) y1--;
    }
    else if (goodcoor(x, y1) == -1) y1=0;
    next = goodcoor(x, y1);
    CmiAssert(next != -1);
    if (next != mype && checkuniq(_n, nb, next)) _n[nb++] = next;
  }
}

int LBTopo_torus2d::get_hop_count(int src,int dest){
	int xpos_src,xpos_dest;
	int ypos_src,ypos_dest;
	int xdist=0;
	int ydist=0;
	
	int xchange;
	if(src > dest){
		xchange = src;
		src = dest;
		dest = xchange;
	}
	
	xpos_src=src%width;
	ypos_src=src/width;

	xpos_dest=dest%width;
	ypos_dest=dest/width;

	xdist = xpos_dest-xpos_src;
	if(xdist<0) xdist=-xdist;
	if((width-xdist) < xdist)
		xdist = width-xdist;
	
	ydist = ypos_dest-ypos_src;
	if(ydist<0) ydist=-ydist;

	int lastpos=(npes-1)%width;
	int otherylen=0;

	if(xpos_src<=lastpos && xpos_dest<=lastpos)
		otherylen=((npes-1)/width)+1-ydist;
	else{
		if(ypos_dest==((npes-1)/width))
			otherylen=((npes-1)/width)+1-ydist;
		else	
			otherylen=((npes-1)/width)-ydist;
	}
	
	if(otherylen < ydist)
		ydist=otherylen;
	
	//added later
	int sdist=0,adist=0,bdist=0,cdist=0,ddist=0;
	
	if(xpos_src>lastpos && xpos_dest>lastpos){
		sdist = xpos_src;
		if((width-sdist) < sdist)
		sdist = width-sdist;

		adist = ((npes-1)/width)-ypos_src;
		if(adist<0) adist=-adist;
		if(ypos_src+1 < adist)
			adist = ypos_src+1;
	
		bdist = 1;

		cdist = ((npes-1)/width)-ypos_dest;
		if(cdist<0) cdist=-cdist;
		if(ypos_dest+1 < cdist)
			cdist = ypos_dest+1;

		ddist = xpos_dest-lastpos;
		if(ddist<0) ddist=-ddist;
		if((width-ddist) < ddist)
			ddist = width-ddist;
	}
	else{
		if(xpos_src>lastpos){
			xchange = src;
			src = dest;
			dest = xchange;
			xpos_src=src%width;
			ypos_src=src/width;
			xpos_dest=dest%width;
			ypos_dest=dest/width;
		}
		adist = ((npes-1)/width)-ypos_src;
		if(adist<0) adist=-adist;
		if(ypos_src+1 < adist)
			adist = ypos_src+1;
	
		if(xpos_dest<=lastpos){
			bdist = xpos_dest-xpos_src;
			if(bdist<0) bdist=-bdist;
			if((lastpos+1-bdist) < bdist)
				bdist = lastpos+1-bdist;

			cdist = ((npes-1)/width)-ypos_dest;
			if(cdist<0) cdist=-cdist;
			if(ypos_dest+1 < cdist)
				cdist = ypos_dest+1;
		
			ddist=0;
		}
		else{
			bdist = lastpos-xpos_src;
			if(bdist<0) bdist=-bdist;
			if((xpos_src+1) < bdist)
				bdist = xpos_src+1;

			cdist = ((npes-1)/width)-ypos_dest;
			if(cdist<0) cdist=-cdist;
			if(ypos_dest+1 < cdist)
				cdist = ypos_dest+1;

			ddist = xpos_dest-lastpos;
			if(ddist<0) ddist=-ddist;
			if((width-ddist) < ddist)
				ddist = width-ddist;
		}
	}
	
	if((sdist+adist+bdist+cdist+ddist) < (xdist+ydist))
		return (sdist+adist+bdist+cdist+ddist);
	else
		return (xdist+ydist);

}

//  TORUS 3D

LBTOPO_MACRO(LBTopo_torus3d)

LBTopo_torus3d::LBTopo_torus3d(int p): LBTopology(p) 
{
  width = 1;
  while ( (width+1) * (width+1) * (width+1) <= npes) width++;
  if (width * width * width < npes) width++;
}

int LBTopo_torus3d::max_neighbors()
{
  return 6;
}

int LBTopo_torus3d::goodcoor(int x, int y, int z)
{
  if (x<0 || x>=width) return -1;
  if (y<0 || y>=width) return -1;
  if (z<0 || z>=width) return -1;
  int next = x*width*width + y*width + z;
  if (next<npes && next>=0) return next;
  return -1;
}

void LBTopo_torus3d::neighbors(int mype, int* _n, int &nb)
{

  int x = mype/(width*width);
  int k = mype%(width*width);
  int y = k/width;
  int z = k%width;
  int next;
  nb=0;
  for (int i=-1; i<=1; i+=2) {
    int x1 = x+i;
    if (x1 == -1) {
      x1 = width-1;
      while (goodcoor(x1, y, z)==-1) x1--;
    }
    else if (goodcoor(x1, y, z) == -1) x1=0;
    next = goodcoor(x1, y, z);
    CmiAssert(next != -1);
    if (next != mype && checkuniq(_n, nb, next)) _n[nb++] = next;

    int y1 = y+i;
    if (y1 == -1) {
      y1 = width-1;
      while (goodcoor(x, y1, z)==-1) y1--;
    }
    else if (goodcoor(x, y1, z) == -1) y1=0;
    next = goodcoor(x, y1, z);
    CmiAssert(next != -1);
    if (next != mype && checkuniq(_n, nb, next)) _n[nb++] = next;

    int z1 = z+i;
    if (z1 == -1) {
      z1 = width-1;
      while (goodcoor(x, y, z1)==-1) z1--;
    }
    else if (goodcoor(x, y, z1) == -1) z1=0;
    next = goodcoor(x, y, z1);
    CmiAssert(next != -1);
    if (next != mype && checkuniq(_n, nb, next)) _n[nb++] = next;
  }
}

/*
// Works only for perfect cube number of processor topologies
int LBTopo_torus3d::get_hop_count(int src,int dest){
	
	int x_src = src/(width*width);
  int k_src = src%(width*width);
  int y_src = k_src/width;
  int z_src = k_src%width;

	int x_dest = dest/(width*width);
  int k_dest = dest%(width*width);
  int y_dest = k_dest/width;
  int z_dest = k_dest%width;

	int xdist=0,ydist=0,zdist=0;
	
	//CmiPrintf("just a chk........\n");
	xdist = x_dest-x_src;
	if(xdist<0) xdist=-xdist;
	if((width-xdist) < xdist)
		xdist = width-xdist;

	ydist = y_dest-y_src;
	if(ydist<0) ydist=-ydist;
	if((width-ydist) < ydist)
		ydist = width-ydist;

	zdist = z_dest-z_src;
	if(zdist<0) zdist=-zdist;
	if((width-zdist) < zdist)
		zdist = width-zdist;

	return (xdist+ydist+zdist);

}
*/

//Mesh3D
LBTOPO_MACRO(LBTopo_mesh3d)

LBTopo_mesh3d::LBTopo_mesh3d(int p): LBTopology(p) 
{
  width = 1;
  while ( (width+1) * (width+1) * (width+1) <= npes) width++;
  if (width * width * width < npes) width++;
}

int LBTopo_mesh3d::max_neighbors()
{
  return 6;
}

int LBTopo_mesh3d::goodcoor(int x, int y, int z)
{
  if (x<0 || x>=width) return -1;
  if (y<0 || y>=width) return -1;
  if (z<0 || z>=width) return -1;
  int next = z*width*width + y*width + x;
  if (next<npes && next>=0) return next;
  return -1;
}

void LBTopo_mesh3d::neighbors(int mype, int* _n, int &nb)
{

  int z = mype/(width*width);
  int k = mype%(width*width);
  int y = k/width;
  int x = k%width;
  int next;
  int isNeigh=1;
  nb=0;
  for (int i=-1; i<=1; i+=2) {
    isNeigh=1;
    int x1 = x+i;
    if (x1 == -1) {
      //x1 = width-1;
      x1=x;
      //while (goodcoor(x1, y, z)==-1) x1--;
      isNeigh=0;
    }
    else if (goodcoor(x1, y, z) == -1) { x1=0; isNeigh=0; }
    next = goodcoor(x1, y, z);
    CmiAssert(next != -1);
    if (next != mype && isNeigh && checkuniq(_n, nb, next)) _n[nb++] = next;

    isNeigh=1;
    int y1 = y+i;
    if (y1 == -1) {
      //y1 = width-1;
      //while (goodcoor(x, y1, z)==-1) y1--;
      y1=y;
      isNeigh=0;
    }
    else if (goodcoor(x, y1, z) == -1) { y1=0; isNeigh=0; }
    next = goodcoor(x, y1, z);
    CmiAssert(next != -1);
    if (next != mype && isNeigh && checkuniq(_n, nb, next)) _n[nb++] = next;

    isNeigh=1;
    int z1 = z+i;
    if (z1 == -1) {
      //z1 = width-1;
      //while (goodcoor(x, y, z1)==-1) z1--;
      z1=z;
      isNeigh=0;
    }
    else if (goodcoor(x, y, z1) == -1) { z1=0; isNeigh=0; }
    next = goodcoor(x, y, z1);
    CmiAssert(next != -1);
    if (next != mype && isNeigh && checkuniq(_n, nb, next)) _n[nb++] = next;
  }
}

//  TORUS ND 
//  added by zshao1

template <int dimension>
class LBTopo_torus_nd: public LBTopology {
private:
  // inherited int npes;
  int* Cardinality;
  int VirtualProcessorCount;
  int* TempCo;
private:
  int GetNeighborID(int ProcessorID, int number) {
    CmiAssert(number>=0 && number<max_neighbors());
    CmiAssert(ProcessorID>=0 && ProcessorID<npes);
    get_processor_coordinates(ProcessorID, TempCo);

    int index = number/2;
    int displacement = (number%2)? -1: 1;
    do{
      TempCo[index] = (TempCo[index] + displacement + Cardinality[index]) % Cardinality[index];
      get_processor_id(TempCo, &ProcessorID);
    } while (ProcessorID >= npes);
    return ProcessorID;
  }
public:
  LBTopo_torus_nd(int p): LBTopology(p) /*inherited :npes(p) */ {
    int i;
    CmiAssert(dimension>=1 && dimension<=16);
    CmiAssert(p>=1);

    Cardinality = new int[dimension];
    TempCo = new int[dimension];
    double pp = p;
    for(i=0;i<dimension;i++) {
      Cardinality[i] = (int)ceil(pow(pp,1.0/(dimension-i))-1e-5);
      pp = pp / Cardinality[i];
    }
    VirtualProcessorCount = 1;
    for(i=0;i<dimension;i++) {
      VirtualProcessorCount *= Cardinality[i];
    }
  }
  ~LBTopo_torus_nd() {
    delete[] Cardinality;
    delete[] TempCo;
  }
  virtual int max_neighbors() {
    return dimension*2;
  }
  virtual void neighbors(int mype, int* _n, int &nb) {
    nb = 0;
    for(int i=0;i<dimension*2;i++) {
      _n[nb] = GetNeighborID(mype, i);
      if (_n[nb]!=mype && (nb==0 || _n[nb-1]!=_n[nb]) ) nb++;
    }
  }
  virtual int get_dimension() {
    return dimension;
  }
  virtual bool get_processor_coordinates(int processor_id, int* processor_coordinates) {
    CmiAssert(processor_id>=0 && processor_id<VirtualProcessorCount);
    CmiAssert( processor_coordinates != NULL );
    for(int i=0;i<dimension;i++) {
      processor_coordinates[i] = processor_id % Cardinality[i];
      processor_id = processor_id / Cardinality[i];
    }
    return true;
  }
  virtual bool get_processor_id(const int* processor_coordinates, int* processor_id) {
    int i;
    CmiAssert( processor_coordinates != NULL );
    CmiAssert( processor_id != NULL );
    for(i=dimension-1;i>=0;i--) 
      CmiAssert( 0<=processor_coordinates[i] && processor_coordinates[i]<Cardinality[i]);
    (*processor_id) = 0;
    for(i=dimension-1;i>=0;i--) {
      (*processor_id) = (*processor_id)* Cardinality[i] + processor_coordinates[i];
    }
    return true;
  }
  //Note: if abs(difference)*2 = cardinality, the difference is set to zero
  virtual bool coordinate_difference(const int* my_coordinates, const int* target_coordinates, int* difference) { 
//    CmiPrintf("[%d] coordiate_difference begin\n", CkMyPe());
    CmiAssert( my_coordinates != NULL);
    CmiAssert( target_coordinates != NULL);
    CmiAssert( difference != NULL);
//    CmiPrintf("[%d] after assert\n", CkMyPe());
    for(int i=0;i<dimension;i++) {
//      CmiPrintf("[%d] coordiate_difference iteration %d\n", i);
      difference[i] = target_coordinates[i] - my_coordinates[i];
      if (abs(difference[i])*2 > Cardinality[i]) {
        difference[i] += (difference[i]>0) ? -Cardinality[i] : Cardinality[i];
      } else if (abs(difference[i])*2 == Cardinality[i]) {
        difference[i] = 0;
      }
    }
//    CmiPrintf("[%d] coordiate_difference just before return\n");
    return true;
  }
  //Note: if abs(difference)*2 = cardinality, the difference is set to zero
  virtual bool coordinate_difference(int my_processor_id, int target_processor_id, int* difference) { 
    CmiAssert( difference != NULL);
    int my_coordinates[dimension];
    int target_coordinates[dimension];
    get_processor_coordinates(my_processor_id, my_coordinates);
    get_processor_coordinates(target_processor_id, target_coordinates);
    coordinate_difference(my_coordinates, target_coordinates, difference);
    return true;
  }
};

typedef LBTopo_torus_nd<1> LBTopo_torus_nd_1;
typedef LBTopo_torus_nd<2> LBTopo_torus_nd_2;
typedef LBTopo_torus_nd<3> LBTopo_torus_nd_3;
typedef LBTopo_torus_nd<4> LBTopo_torus_nd_4;
typedef LBTopo_torus_nd<5> LBTopo_torus_nd_5;
typedef LBTopo_torus_nd<6> LBTopo_torus_nd_6;
typedef LBTopo_torus_nd<7> LBTopo_torus_nd_7;
typedef LBTopo_torus_nd<8> LBTopo_torus_nd_8;
typedef LBTopo_torus_nd<9> LBTopo_torus_nd_9;
typedef LBTopo_torus_nd<10> LBTopo_torus_nd_10;

LBTOPO_MACRO(LBTopo_torus_nd_1)
LBTOPO_MACRO(LBTopo_torus_nd_2)
LBTOPO_MACRO(LBTopo_torus_nd_3)
LBTOPO_MACRO(LBTopo_torus_nd_4)
LBTOPO_MACRO(LBTopo_torus_nd_5)
LBTOPO_MACRO(LBTopo_torus_nd_6)
LBTOPO_MACRO(LBTopo_torus_nd_7)
LBTOPO_MACRO(LBTopo_torus_nd_8)
LBTOPO_MACRO(LBTopo_torus_nd_9)
LBTOPO_MACRO(LBTopo_torus_nd_10)

//  TORUS ND  and SMP Awareness
//  added by Yanhua Sun 

//#define YHDEBUG

template <int dimension>
class LBTopo_torus_nd_smp: public LBTopology {
private:
  // inherited int npes;
  int* Cardinality;
  int  VirtualNodeCount;
  int* TempCo;
  int  ppn;
  int  NumOfNodes;
private:
  int GetNeighborID(int ProcessorID, int number) {
    CmiAssert(number>=0 && number<max_neighbors());
    CmiAssert(ProcessorID>=0 && ProcessorID<npes);
    int neighborId; 
    int nodeId = CmiPhysicalNodeID(ProcessorID);
    
    get_node_coordinates(nodeId, TempCo);

    int index = number/2;
    int displacement = (number%2)? -1: 1;
    do{
      TempCo[index] = (TempCo[index] + displacement + Cardinality[index]) % Cardinality[index];
      get_node_id(TempCo, &nodeId);
    } while (nodeId >= NumOfNodes);
    neighborId = CmiGetFirstPeOnPhysicalNode(nodeId);
    return neighborId;
  }
public:
  LBTopo_torus_nd_smp(int p): LBTopology(p) /*inherited :npes(p) */ {
    int i;
    CmiAssert(dimension>=1 && dimension<=32);
    CmiAssert(p>=1);

    ppn = CmiNumPesOnPhysicalNode(0);
    NumOfNodes = CmiNumPhysicalNodes();

    Cardinality = new int[dimension];
    TempCo = new int[dimension];
    double pp = NumOfNodes;
    for(i=0;i<dimension;i++) {
      Cardinality[i] = (int)ceil(pow(pp,1.0/(dimension-i))-1e-5);
      pp = pp / Cardinality[i];
    }
    VirtualNodeCount = 1;
    for(i=0;i<dimension;i++) {
      VirtualNodeCount *= Cardinality[i];
    }
#ifdef YHDEBUG
    CmiPrintf(" ppn=%d, NumOfNodes=%d\n", ppn, NumOfNodes);
#endif
  }
  ~LBTopo_torus_nd_smp() {
    delete[] Cardinality;
    delete[] TempCo;
  }
  virtual int max_neighbors() {
    return (dimension+ppn)*2;
  }
  virtual void neighbors(int mype, int* _n, int &nb) {
    nb = 0;
    int *nodePeList; 
    int numpes;
    int rank = CmiPhysicalRank(mype);
    int node = CmiPhysicalNodeID(mype);
    int _ppn_ = CmiNumPesOnPhysicalNode(node);
    CmiGetPesOnPhysicalNode(node, &nodePeList, &numpes); 
#ifdef YHDEBUG
    CmiPrintf(" PE[%d] ppn=%d, NumOfNodes=%d, rank=%d, node=%d, numpes=%d\n", mype, _ppn_, NumOfNodes, rank, node, numpes);
#endif   
    for(int i=0; i<numpes; i++)
    {
        int _pid = nodePeList[i];
        if(_pid != mype)
        {
             _n[nb] = _pid;
             nb++;
        }
    }

    /* for inter-node communication */
    if(mype == CmiGetFirstPeOnPhysicalNode(node))
    {
        for(int j=0; j<dimension*2; j++)
        {
            //_n[nb] = (mype+1)%npes;//GetNeighborID(mype, j);
            _n[nb] = GetNeighborID(mype, j);
            /* the first processors in other nodes */
            if (_n[nb]!=mype && (nb==0 || _n[nb-1]!=_n[nb]) ) nb++;
        }
    }

#ifdef YHDEBUG
  CmiPrintf(" [%d] neighbor = %d ppn=%d, NumOfNodes=%d, rank=%d, node=%d, numpes=%d\n", mype, nb, _ppn_, NumOfNodes, rank, node, numpes);
#endif
  }
  virtual int get_dimension() {
    return dimension;
  }
  virtual bool get_node_coordinates(int node_id, int* node_coordinates) {
    CmiAssert(node_id>=0 && node_id<VirtualNodeCount);
    CmiAssert( node_coordinates != NULL );
    for(int i=0;i<dimension;i++) {
      node_coordinates[i] = node_id % Cardinality[i];
      node_id = node_id / Cardinality[i];
    }
    return true;
  }
  virtual bool get_node_id(const int* node_coordinates, int* node_id) {
    int i;
    CmiAssert( node_coordinates != NULL );
    CmiAssert( node_id != NULL );
    for(i=dimension-1;i>=0;i--) 
      CmiAssert( 0<=node_coordinates[i] && node_coordinates[i]<Cardinality[i]);
    (*node_id) = 0;
    for(i=dimension-1;i>=0;i--) {
      (*node_id) = (*node_id)* Cardinality[i] + node_coordinates[i];
    }
    return true;
  }
  //Note: if abs(difference)*2 = cardinality, the difference is set to zero
  virtual bool coordinate_difference(const int* my_coordinates, const int* target_coordinates, int* difference) { 
    CmiAssert( my_coordinates != NULL);
    CmiAssert( target_coordinates != NULL);
    CmiAssert( difference != NULL);
    for(int i=0;i<dimension;i++) {
      difference[i] = target_coordinates[i] - my_coordinates[i];
      if (abs(difference[i])*2 > Cardinality[i]) {
        difference[i] += (difference[i]>0) ? -Cardinality[i] : Cardinality[i];
      } else if (abs(difference[i])*2 == Cardinality[i]) {
        difference[i] = 0;
      }
    }
    return true;
  }
  //Note: if abs(difference)*2 = cardinality, the difference is set to zero
  virtual bool coordinate_difference(int my_processor_id, int target_processor_id, int* difference) { 
    CmiAssert( difference != NULL);
    int my_coordinates[dimension];
    int target_coordinates[dimension];
    get_processor_coordinates(my_processor_id, my_coordinates);
    get_processor_coordinates(target_processor_id, target_coordinates);
    coordinate_difference(my_coordinates, target_coordinates, difference);
    return true;
  }
};


typedef LBTopo_torus_nd_smp<1> LBTopo_torus_nd_smp_1;
typedef LBTopo_torus_nd_smp<2> LBTopo_torus_nd_smp_2;
typedef LBTopo_torus_nd_smp<3> LBTopo_torus_nd_smp_3;
typedef LBTopo_torus_nd_smp<4> LBTopo_torus_nd_smp_4;
typedef LBTopo_torus_nd_smp<5> LBTopo_torus_nd_smp_5;
typedef LBTopo_torus_nd_smp<6> LBTopo_torus_nd_smp_6;
typedef LBTopo_torus_nd_smp<7> LBTopo_torus_nd_smp_7;
typedef LBTopo_torus_nd_smp<8> LBTopo_torus_nd_smp_8;
typedef LBTopo_torus_nd_smp<9> LBTopo_torus_nd_smp_9;
typedef LBTopo_torus_nd_smp<10> LBTopo_torus_nd_smp_10;

LBTOPO_MACRO(LBTopo_torus_nd_smp_1)
LBTOPO_MACRO(LBTopo_torus_nd_smp_2)
LBTOPO_MACRO(LBTopo_torus_nd_smp_3)
LBTOPO_MACRO(LBTopo_torus_nd_smp_4)
LBTOPO_MACRO(LBTopo_torus_nd_smp_5)
LBTOPO_MACRO(LBTopo_torus_nd_smp_6)
LBTOPO_MACRO(LBTopo_torus_nd_smp_7)
LBTOPO_MACRO(LBTopo_torus_nd_smp_8)
LBTOPO_MACRO(LBTopo_torus_nd_smp_9)
LBTOPO_MACRO(LBTopo_torus_nd_smp_10)


//Torus ND with unequal number of processors in each dimension
/***************************************************************/
template <int dimension>
class LBTopo_itorus_nd: public LBTopology {
private:
	int *dim;
	int *tempCoor;
	
public:
	LBTopo_itorus_nd(int p): LBTopology(p) {
  	CmiPrintf("Irregular torus created\n");
  	dim = new int[dimension];
		tempCoor = new int[dimension];

		int i=0;
  	char *lbcopy = strdup(_lbtopo);
  	char *ptr = strchr(lbcopy, ':');
    if (ptr==NULL) {
      free(lbcopy);
      return;
    }
  	ptr = strtok(ptr+1, ",");
  	while (ptr) {
			dim[i]=atoi(ptr);
			i++;
			ptr = strtok(NULL, ",");
  	}
		CmiAssert(dimension==i);
		
		int procs=1;
		for(i=0;i<dimension;i++)
			procs*=dim[i];
    CmiAssert(dimension>=1 && dimension<=16);
    CmiAssert(p>=1);
		CmiAssert(procs==p);
    free(lbcopy);
  }
	
  ~LBTopo_itorus_nd() {
  	delete[] dim;
		delete[] tempCoor;
	}
	
  virtual int max_neighbors() {
    return dimension*2;
  }
	
	virtual void neighbors(int mype, int* _n, int &nb) {
    nb = 0;
    for(int i=0;i<dimension*2;i++) {
      _n[nb] = GetNeighborID(mype, i);
      if (_n[nb]!=mype && (nb==0 || _n[nb-1]!=_n[nb]) ) nb++;
    }
  }

	int GetNeighborID(int ProcessorID, int number) {
    CmiAssert(number>=0 && number<max_neighbors());
    CmiAssert(ProcessorID>=0 && ProcessorID<npes);
    get_processor_coordinates(ProcessorID, tempCoor);

    int index = number/2;
    int displacement = (number%2)? -1: 1;
   // do{
		tempCoor[index] = (tempCoor[index] + displacement + dim[index]) % dim[index];
		get_processor_id(tempCoor, &ProcessorID);
    //} while (ProcessorID >= npes);
    return ProcessorID;
  }

	virtual bool get_processor_coordinates(int processor_id, int* processor_coordinates) {
    CmiAssert(processor_id>=0 && processor_id<npes);
    CmiAssert(processor_coordinates != NULL );
    for(int i=0;i<dimension;i++) {
      processor_coordinates[i] = processor_id % dim[i];
      processor_id = processor_id / dim[i];
    }
    return true;
  }

	virtual bool get_processor_id(const int* processor_coordinates, int* processor_id) {
    int i;
    CmiAssert( processor_coordinates != NULL );
    CmiAssert( processor_id != NULL );
    for(i=dimension-1;i>=0;i--) 
      CmiAssert( 0<=processor_coordinates[i] && processor_coordinates[i]<dim[i]);
    (*processor_id) = 0;
    for(i=dimension-1;i>=0;i--) {
      (*processor_id) = (*processor_id)* dim[i] + processor_coordinates[i];
    }
    return true;
  }
};


typedef LBTopo_itorus_nd<1> LBTopo_itorus_nd_1;
typedef LBTopo_itorus_nd<2> LBTopo_itorus_nd_2;
typedef LBTopo_itorus_nd<3> LBTopo_itorus_nd_3;
typedef LBTopo_itorus_nd<4> LBTopo_itorus_nd_4;
typedef LBTopo_itorus_nd<5> LBTopo_itorus_nd_5;
typedef LBTopo_itorus_nd<6> LBTopo_itorus_nd_6;
typedef LBTopo_itorus_nd<7> LBTopo_itorus_nd_7;

LBTOPO_MACRO(LBTopo_itorus_nd_1)
LBTOPO_MACRO(LBTopo_itorus_nd_2)
LBTOPO_MACRO(LBTopo_itorus_nd_3)
LBTOPO_MACRO(LBTopo_itorus_nd_4)
LBTOPO_MACRO(LBTopo_itorus_nd_5)
LBTOPO_MACRO(LBTopo_itorus_nd_6)
LBTOPO_MACRO(LBTopo_itorus_nd_7)


/******************************************************************/
//Mesh ND with unequal number of processors in each dimension
/***************************************************************/
template <int dimension>
class LBTopo_imesh_nd: public LBTopology {
private:
	int *dim;
	int *tempCoor;
	
public:
	LBTopo_imesh_nd(int p): LBTopology(p) {
  	CmiPrintf("Irregular mesh created\n");
  	dim = new int[dimension];
	tempCoor = new int[dimension];

	int i=0;
  	char *lbcopy = strdup(_lbtopo);
  	char *ptr = strchr(lbcopy, ':');
  	if (ptr==NULL) 
	  {
	    delete [] dim;
	    delete [] tempCoor;
	    free(lbcopy);
	    return;
	  }
  	ptr = strtok(ptr+1, ",");
  	while (ptr) {
	  dim[i]=atoi(ptr);
	  i++;
	  ptr = strtok(NULL, ",");
  	}
	CmiAssert(dimension==i);
	
	//char *ptr2=strchr(_lbtopo,':');
	//*ptr2='\0';
	int procs=1;
	for(i=0;i<dimension;i++)
	  procs*=dim[i];
          CmiAssert(dimension>=1 && dimension<=16);
          CmiAssert(p>=1);
	  CmiAssert(procs==p);
	  free(lbcopy);
  }
	
  ~LBTopo_imesh_nd() {
  	delete[] dim;
		delete[] tempCoor;
	}
	
  virtual int max_neighbors() {
    return dimension*2;
  }
	
  virtual void neighbors(int mype, int* _n, int &nb) {
    nb = 0;
    for(int i=0;i<dimension*2;i++) {
      _n[nb] = GetNeighborID(mype, i);
      if (_n[nb]!=mype && (nb==0 || _n[nb-1]!=_n[nb]) ) nb++;
    }
    /*CmiPrintf("Nhs[%d]:",mype);
    for(int i=0;i<nb;i++)
      CmiPrintf("%d ",_n[i]);
    CmiPrintf("\n");*/
  }

	int GetNeighborID(int ProcessorID, int number) {
        CmiAssert(number>=0 && number<max_neighbors());
        CmiAssert(ProcessorID>=0 && ProcessorID<npes);
        get_processor_coordinates(ProcessorID, tempCoor);

        int index = number/2;
        int displacement = (number%2)? -1: 1;
   // do{
	    if((tempCoor[index]==0 && displacement==-1) || (tempCoor[index]==dim[index]-1 && displacement==1)){
        }
        else{
	     tempCoor[index] = (tempCoor[index] + displacement + dim[index]) % dim[index];
	     get_processor_id(tempCoor, &ProcessorID);
	    }
    //} while (ProcessorID >= npes);
    return ProcessorID;
  }

	virtual bool get_processor_coordinates(int processor_id, int* processor_coordinates) {
    CmiAssert(processor_id>=0 && processor_id<npes);
    CmiAssert(processor_coordinates != NULL );
    for(int i=0;i<dimension;i++) {
      processor_coordinates[i] = processor_id % dim[i];
      processor_id = processor_id / dim[i];
    }
    return true;
  }

	virtual bool get_processor_id(const int* processor_coordinates, int* processor_id) {
    int i;
    CmiAssert( processor_coordinates != NULL );
    CmiAssert( processor_id != NULL );
    for(i=dimension-1;i>=0;i--) 
      CmiAssert( 0<=processor_coordinates[i] && processor_coordinates[i]<dim[i]);
    (*processor_id) = 0;
    for(i=dimension-1;i>=0;i--) {
      (*processor_id) = (*processor_id)* dim[i] + processor_coordinates[i];
    }
    return true;
  }
};


typedef LBTopo_imesh_nd<1> LBTopo_imesh_nd_1;
typedef LBTopo_imesh_nd<2> LBTopo_imesh_nd_2;
typedef LBTopo_imesh_nd<3> LBTopo_imesh_nd_3;
typedef LBTopo_imesh_nd<4> LBTopo_imesh_nd_4;
typedef LBTopo_imesh_nd<5> LBTopo_imesh_nd_5;
typedef LBTopo_imesh_nd<6> LBTopo_imesh_nd_6;
typedef LBTopo_imesh_nd<7> LBTopo_imesh_nd_7;

LBTOPO_MACRO(LBTopo_imesh_nd_1)
LBTOPO_MACRO(LBTopo_imesh_nd_2)
LBTOPO_MACRO(LBTopo_imesh_nd_3)
LBTOPO_MACRO(LBTopo_imesh_nd_4)
LBTOPO_MACRO(LBTopo_imesh_nd_5)
LBTOPO_MACRO(LBTopo_imesh_nd_6)
LBTOPO_MACRO(LBTopo_imesh_nd_7)


// dense graph with connectivity of square root processor number

LBTOPO_MACRO(LBTopo_graph)

int LBTopo_graph::max_neighbors()
{
  return (int)(sqrt(1.0*CmiNumPes())+0.5);
}

extern "C" void gengraph(int, int, int, int *, int *, int);

void LBTopo_graph::neighbors(int mype, int* na, int &nb)
{
  gengraph(CmiNumPes(), (int)(sqrt(1.0*CmiNumPes())+0.5), 234, na, &nb, 0);
}

/* add by Yanhua Aug-2010*/
template <int dimension>
class LBTopo_graph_nc: public LBTopology {

public:
    LBTopo_graph_nc(int p): LBTopology(p) {}
    int max_neighbors()
    {
        return dimension + 1; 
    }

    void neighbors(int mype, int* na, int &nb)
    {
#if CMK_NODE_QUEUE_AVAILABLE
        gengraph(CmiNumNodes(), dimension, 234, na, &nb, 0);
#else
        gengraph(CmiNumPes(), dimension, 234, na, &nb, 0);
#endif
    }

};
typedef LBTopo_graph_nc<2> LBTopo_graph_nc_2;
typedef LBTopo_graph_nc<3> LBTopo_graph_nc_3;
typedef LBTopo_graph_nc<4> LBTopo_graph_nc_4;
typedef LBTopo_graph_nc<5> LBTopo_graph_nc_5;
typedef LBTopo_graph_nc<6> LBTopo_graph_nc_6;
typedef LBTopo_graph_nc<7> LBTopo_graph_nc_7;
typedef LBTopo_graph_nc<8> LBTopo_graph_nc_8;
typedef LBTopo_graph_nc<9> LBTopo_graph_nc_9;
typedef LBTopo_graph_nc<10> LBTopo_graph_nc_10;
typedef LBTopo_graph_nc<20> LBTopo_graph_nc_20;

LBTOPO_MACRO(LBTopo_graph_nc_2)
LBTOPO_MACRO(LBTopo_graph_nc_3)
LBTOPO_MACRO(LBTopo_graph_nc_4)
LBTOPO_MACRO(LBTopo_graph_nc_5)
LBTOPO_MACRO(LBTopo_graph_nc_6)
LBTOPO_MACRO(LBTopo_graph_nc_7)
LBTOPO_MACRO(LBTopo_graph_nc_8)
LBTOPO_MACRO(LBTopo_graph_nc_9)
LBTOPO_MACRO(LBTopo_graph_nc_10)
LBTOPO_MACRO(LBTopo_graph_nc_20)


/* Centralized  balancer, one processor has the neighbors of all other processors, while the other ones only have one neighbor, the centralized processor */

 

// complete graph

class LBTopo_complete: public LBTopology {
public:
  LBTopo_complete(int p): LBTopology(p) {}
  int max_neighbors() {
    return npes - 1;
  }
  void neighbors(int mype, int* _n, int &nb) {
    nb = 0;
    for (int i=0; i<npes; i++)  if (mype != i) _n[nb++] = i;
  }
};

LBTOPO_MACRO(LBTopo_complete)

//   k-ary tree

template <int k>
class LBTopo_karytree: public LBTopology {
public:
  LBTopo_karytree(int p): LBTopology(p) {}
  virtual int max_neighbors() {
    return k+1;     // parent + children
  }
  virtual void neighbors(int mype, int* _n, int &nb) {
    nb = 0;
    if (mype!=0) _n[nb++] = (mype-1)/k;
    int firstchild = mype*k+1;
    for (int i=0; i<k; i++)
      if (firstchild+i < npes) _n[nb++] = firstchild+i;
  }
};

typedef LBTopo_karytree<2> LBTopo_2_arytree;
typedef LBTopo_karytree<3> LBTopo_3_arytree;
typedef LBTopo_karytree<4> LBTopo_4_arytree;
typedef LBTopo_karytree<128> LBTopo_128_arytree;
typedef LBTopo_karytree<512> LBTopo_512_arytree;

LBTOPO_MACRO(LBTopo_2_arytree)
LBTOPO_MACRO(LBTopo_3_arytree)
LBTOPO_MACRO(LBTopo_4_arytree)
LBTOPO_MACRO(LBTopo_128_arytree)
LBTOPO_MACRO(LBTopo_512_arytree)

//

class LBTopoMap {
public:
  const char *name;
  LBtopoFn fn;
  LBTopoMap(const char *s, LBtopoFn f): name(s), fn(f) {}
  LBTopoMap(const LBTopoMap &p);		// You don't want to copy
  void operator=(const LBTopoMap &p);		// You don't want to copy
};

class LBTopoVec {
  CkVec<LBTopoMap *> lbTopos;
public:
  LBTopoVec() {
    // register all topos
    lbTopos.push_back(new LBTopoMap("ring", createLBTopo_ring));
    lbTopos.push_back(new LBTopoMap("torus2d", createLBTopo_torus2d));
    lbTopos.push_back(new LBTopoMap("torus3d", createLBTopo_torus3d));
    lbTopos.push_back(new LBTopoMap("mesh3d", createLBTopo_mesh3d));
    lbTopos.push_back(new LBTopoMap("torus_nd_1", createLBTopo_torus_nd_1));
    lbTopos.push_back(new LBTopoMap("torus_nd_2", createLBTopo_torus_nd_2));
    lbTopos.push_back(new LBTopoMap("torus_nd_3", createLBTopo_torus_nd_3));
    lbTopos.push_back(new LBTopoMap("torus_nd_4", createLBTopo_torus_nd_4));
    lbTopos.push_back(new LBTopoMap("torus_nd_5", createLBTopo_torus_nd_5));
    lbTopos.push_back(new LBTopoMap("torus_nd_6", createLBTopo_torus_nd_6));
    lbTopos.push_back(new LBTopoMap("torus_nd_7", createLBTopo_torus_nd_7));
    lbTopos.push_back(new LBTopoMap("torus_nd_8", createLBTopo_torus_nd_8));
    lbTopos.push_back(new LBTopoMap("torus_nd_9", createLBTopo_torus_nd_9));
    lbTopos.push_back(new LBTopoMap("torus_nd_10", createLBTopo_torus_nd_10));
    lbTopos.push_back(new LBTopoMap("torus_nd_smp_1", createLBTopo_torus_nd_smp_1));
    lbTopos.push_back(new LBTopoMap("torus_nd_smp_2", createLBTopo_torus_nd_smp_2));
    lbTopos.push_back(new LBTopoMap("torus_nd_smp_3", createLBTopo_torus_nd_smp_3));
    lbTopos.push_back(new LBTopoMap("torus_nd_smp_4", createLBTopo_torus_nd_smp_4));
    lbTopos.push_back(new LBTopoMap("torus_nd_smp_5", createLBTopo_torus_nd_smp_5));
    lbTopos.push_back(new LBTopoMap("torus_nd_smp_6", createLBTopo_torus_nd_smp_6));
    lbTopos.push_back(new LBTopoMap("torus_nd_smp_7", createLBTopo_torus_nd_smp_7));
    lbTopos.push_back(new LBTopoMap("torus_nd_smp_8", createLBTopo_torus_nd_smp_8));
    lbTopos.push_back(new LBTopoMap("torus_nd_smp_9", createLBTopo_torus_nd_smp_9));
    lbTopos.push_back(new LBTopoMap("torus_nd_smp_10", createLBTopo_torus_nd_smp_10));
    lbTopos.push_back(new LBTopoMap("itorus_nd_1", createLBTopo_itorus_nd_1));
    lbTopos.push_back(new LBTopoMap("itorus_nd_2", createLBTopo_itorus_nd_2));
    lbTopos.push_back(new LBTopoMap("itorus_nd_3", createLBTopo_itorus_nd_3));
    lbTopos.push_back(new LBTopoMap("itorus_nd_4", createLBTopo_itorus_nd_4));
    lbTopos.push_back(new LBTopoMap("itorus_nd_5", createLBTopo_itorus_nd_5));
    lbTopos.push_back(new LBTopoMap("itorus_nd_6", createLBTopo_itorus_nd_6));
    lbTopos.push_back(new LBTopoMap("itorus_nd_7", createLBTopo_itorus_nd_7));
    lbTopos.push_back(new LBTopoMap("imesh_nd_1", createLBTopo_imesh_nd_1));
    lbTopos.push_back(new LBTopoMap("imesh_nd_2", createLBTopo_imesh_nd_2));
    lbTopos.push_back(new LBTopoMap("imesh_nd_3", createLBTopo_imesh_nd_3));
    lbTopos.push_back(new LBTopoMap("imesh_nd_4", createLBTopo_imesh_nd_4));
    lbTopos.push_back(new LBTopoMap("imesh_nd_5", createLBTopo_imesh_nd_5));
    lbTopos.push_back(new LBTopoMap("imesh_nd_6", createLBTopo_imesh_nd_6));
    lbTopos.push_back(new LBTopoMap("imesh_nd_7", createLBTopo_imesh_nd_7));
    lbTopos.push_back(new LBTopoMap("graph", createLBTopo_graph));
    lbTopos.push_back(new LBTopoMap("graph_nc_2", createLBTopo_graph_nc_2));
    lbTopos.push_back(new LBTopoMap("graph_nc_3", createLBTopo_graph_nc_3));
    lbTopos.push_back(new LBTopoMap("graph_nc_4", createLBTopo_graph_nc_4));
    lbTopos.push_back(new LBTopoMap("graph_nc_5", createLBTopo_graph_nc_5));
    lbTopos.push_back(new LBTopoMap("graph_nc_6", createLBTopo_graph_nc_6));
    lbTopos.push_back(new LBTopoMap("graph_nc_7", createLBTopo_graph_nc_7));
    lbTopos.push_back(new LBTopoMap("graph_nc_8", createLBTopo_graph_nc_8));
    lbTopos.push_back(new LBTopoMap("graph_nc_9", createLBTopo_graph_nc_9));
    lbTopos.push_back(new LBTopoMap("graph_nc_10", createLBTopo_graph_nc_10));
    lbTopos.push_back(new LBTopoMap("graph_nc_20", createLBTopo_graph_nc_20));
    lbTopos.push_back(new LBTopoMap("complete", createLBTopo_complete));
    lbTopos.push_back(new LBTopoMap("2_arytree", createLBTopo_2_arytree));
    lbTopos.push_back(new LBTopoMap("3_arytree", createLBTopo_3_arytree));
    lbTopos.push_back(new LBTopoMap("4_arytree", createLBTopo_4_arytree));
    lbTopos.push_back(new LBTopoMap("128_arytree", createLBTopo_128_arytree));
    lbTopos.push_back(new LBTopoMap("512_arytree", createLBTopo_512_arytree));
    lbTopos.push_back(new LBTopoMap("smp_n_1", createLBTopo_smp_n_1));
    lbTopos.push_back(new LBTopoMap("smp_n_2", createLBTopo_smp_n_2));
    lbTopos.push_back(new LBTopoMap("smp_n_3", createLBTopo_smp_n_3));
    lbTopos.push_back(new LBTopoMap("smp_n_4", createLBTopo_smp_n_4));
    lbTopos.push_back(new LBTopoMap("smp_n_5", createLBTopo_smp_n_5));
    lbTopos.push_back(new LBTopoMap("smp_n_6", createLBTopo_smp_n_6));
    lbTopos.push_back(new LBTopoMap("smp_n_7", createLBTopo_smp_n_7));
    lbTopos.push_back(new LBTopoMap("smp_n_8", createLBTopo_smp_n_8));
    lbTopos.push_back(new LBTopoMap("smp_n_9", createLBTopo_smp_n_9));
    lbTopos.push_back(new LBTopoMap("smp_n_10", createLBTopo_smp_n_10));
  }
  ~LBTopoVec() {
    for (int i=0; i<lbTopos.length(); i++)
      delete lbTopos[i];
  }
  void push_back(LBTopoMap *map) { lbTopos.push_back(map); }
  int length() { return lbTopos.length(); }
  LBTopoMap * operator[](size_t n) { return lbTopos[n]; }
  void print() {
    for (int i=0; i<lbTopos.length(); i++) {
      CmiPrintf("  %s\n", lbTopos[i]->name);
    }
  }
};

static LBTopoVec lbTopoMap;

extern "C"
LBtopoFn LBTopoLookup(char *name)
{
  for (int i=0; i<lbTopoMap.length(); i++) {
    if (strcmp(name, lbTopoMap[i]->name)==0) return lbTopoMap[i]->fn;
  }
  return NULL;
}

// C wrapper functions
extern "C" void getTopoNeighbors(void *topo, int myid, int* narray, int *n)
{
  ((LBTopology*)topo)->neighbors(myid, narray, *n);
}

extern "C" int getTopoMaxNeighbors(void *topo)
{
  return ((LBTopology*)topo)->max_neighbors();
}

extern "C" void printoutTopo()
{
  lbTopoMap.print();
}

