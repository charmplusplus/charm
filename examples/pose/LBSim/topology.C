/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _TOPOLOGY_C_
#define _TOPOLOGY_C_

#include <math.h>

//#include "LBDatabase.h"
//#include "charm.h"
#include "cklists.h"
#include "topology.h"
#include "typedefs.h"

// ring

LBTOPO_MACRO(LBTopo_ring);

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

//  TORUS 2D

LBTOPO_MACRO(LBTopo_torus2d);

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
    //CmiAssert(next != -1);
    if (next != mype && checkuniq(_n, nb, next)) _n[nb++] = next;

    int y1 = y+i;
    if (y1 == -1) {
      y1 = width-1;
      while (goodcoor(x, y1)==-1) y1--;
    }
    else if (goodcoor(x, y1) == -1) y1=0;
    next = goodcoor(x, y1);
    //CmiAssert(next != -1);
    if (next != mype && checkuniq(_n, nb, next)) _n[nb++] = next;
  }
}


//  TORUS 3D

LBTOPO_MACRO(LBTopo_torus3d);

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
    //CmiAssert(next != -1);
    if (next != mype && checkuniq(_n, nb, next)) _n[nb++] = next;

    int y1 = y+i;
    if (y1 == -1) {
      y1 = width-1;
      while (goodcoor(x, y1, z)==-1) y1--;
    }
    else if (goodcoor(x, y1, z) == -1) y1=0;
    next = goodcoor(x, y1, z);
    //CmiAssert(next != -1);
    if (next != mype && checkuniq(_n, nb, next)) _n[nb++] = next;

    int z1 = z+i;
    if (z1 == -1) {
      z1 = width-1;
      while (goodcoor(x, y, z1)==-1) z1--;
    }
    else if (goodcoor(x, y, z1) == -1) z1=0;
    next = goodcoor(x, y, z1);
    //CmiAssert(next != -1);
    if (next != mype && checkuniq(_n, nb, next)) _n[nb++] = next;
  }
}

// dense graph

LBTOPO_MACRO(LBTopo_graph);

int LBTopo_graph::max_neighbors()
{
  return (int)(1.0 + (log10(1.0*CmiNumPes())/log10(2.0)));
}

extern "C" void gengraph(int, int, int, int *, int *, int, int);

void LBTopo_graph::neighbors(int mype, int* na, int &nb)
{
  gengraph(CmiNumPes(), (int)(1.0 + (log10(1.0*CmiNumPes())/log10(2.0))), 234, na, &nb, 0,mype);
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
  LBTopo_torus_nd(int p): LBTopology(p) /*inherited :npes(p) */ 
  {  int i;
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
//    CkPrintf("[%d] coordiate_difference begin\n", CkMyPe());
    CmiAssert( my_coordinates != NULL);
    CmiAssert( target_coordinates != NULL);
    CmiAssert( difference != NULL);
//    CkPrintf("[%d] after assert\n", CkMyPe());
    for(int i=0;i<dimension;i++) {
//      CkPrintf("[%d] coordiate_difference iteration %d\n", i);
      difference[i] = target_coordinates[i] - my_coordinates[i];
      if (abs(difference[i])*2 > Cardinality[i]) {
        difference[i] += (difference[i]>0) ? -Cardinality[i] : Cardinality[i];
      } else if (abs(difference[i])*2 == Cardinality[i]) {
        difference[i] = 0;
      }
    }
//    CkPrintf("[%d] coordiate_difference just before return\n");
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

LBTOPO_MACRO(LBTopo_torus_nd_1);
LBTOPO_MACRO(LBTopo_torus_nd_2);
LBTOPO_MACRO(LBTopo_torus_nd_3);
LBTOPO_MACRO(LBTopo_torus_nd_4);
LBTOPO_MACRO(LBTopo_torus_nd_5);
LBTOPO_MACRO(LBTopo_torus_nd_6);
LBTOPO_MACRO(LBTopo_torus_nd_7);


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

LBTOPO_MACRO(LBTopo_complete);

class LBTopoMap {
public:
  char *name;
  LBtopoFn fn;
  LBTopoMap(char *s, LBtopoFn f): name(s), fn(f) {}
};


class LBTopoVec {
  CkVec<LBTopoMap *> lbTopos;
public:
  LBTopoVec() {
    // register all topos
    lbTopos.push_back(new LBTopoMap("ring", createLBTopo_ring));
    lbTopos.push_back(new LBTopoMap("torus2d", createLBTopo_torus2d));
    lbTopos.push_back(new LBTopoMap("torus3d", createLBTopo_torus3d));
    lbTopos.push_back(new LBTopoMap("torus_nd_1", createLBTopo_torus_nd_1));
    lbTopos.push_back(new LBTopoMap("torus_nd_2", createLBTopo_torus_nd_2));
    lbTopos.push_back(new LBTopoMap("torus_nd_3", createLBTopo_torus_nd_3));
    lbTopos.push_back(new LBTopoMap("torus_nd_4", createLBTopo_torus_nd_4));
    lbTopos.push_back(new LBTopoMap("torus_nd_5", createLBTopo_torus_nd_5));
    lbTopos.push_back(new LBTopoMap("torus_nd_6", createLBTopo_torus_nd_6));
    lbTopos.push_back(new LBTopoMap("torus_nd_7", createLBTopo_torus_nd_7));
    lbTopos.push_back(new LBTopoMap("graph", createLBTopo_graph));
    lbTopos.push_back(new LBTopoMap("complete", createLBTopo_complete));
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
  //printf("maplen: %d\n",lbTopoMap.length());
  for (int i=0; i<lbTopoMap.length(); i++) {
    if (strcmp(name, lbTopoMap[i]->name)==0) return lbTopoMap[i]->fn;
  }
  return NULL;
}

// C wrapper functions
/*extern "C" void getTopoNeighbors(void *topo, int myid, int* narray, int *n)
{
  ((LBTopology*)topo)->neighbors(myid, narray, *n);
}

extern "C" int getTopoMaxNeighbors(void *topo)
{
  return ((LBTopology*)topo)->max_neighbors();
}

extern "C" void printoutTopo()
{
  for (int i=0; i<lbTopoMap.length(); i++) {
    CmiPrintf("  %s\n", lbTopoMap[i]->name);
  }
}
*/

#endif
