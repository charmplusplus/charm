/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _LBTOPOLOGY_H
#define _LBTOPOLOGY_H

#ifdef __cplusplus

class LBTopology {
protected:
  int npes;
public:
  LBTopology(int p): npes(p) {}
  virtual int max_neighbors() = 0;
  virtual void neighbors(int mype, int* _n, int &nb) = 0;
};

#define LBTOPO_MACRO(x) \
  static LBTopology * create##x() { 	\
    return new x(CkNumPes()); 	\
  }

class LBTopo_ring: public LBTopology {
public:
  LBTopo_ring(int p): LBTopology(p) {}
  virtual int max_neighbors();
  virtual void neighbors(int mype, int* _n, int &nb);
};

class LBTopo_mesh2d: public LBTopology {
private:
  int width;
  int goodcoor(int, int);
public:
  LBTopo_mesh2d(int p);
  virtual int max_neighbors();
  virtual void neighbors(int mype, int* _n, int &nb);
};

class LBTopo_mesh3d: public LBTopology {
private:
  int width;
  int goodcoor(int, int, int);
public:
  LBTopo_mesh3d(int p);
  virtual int max_neighbors();
  virtual void neighbors(int mype, int* _n, int &nb);
};

class LBTopo_graph: public LBTopology {
public:
  LBTopo_graph(int p): LBTopology(p) {}
  virtual int max_neighbors();
  virtual void neighbors(int mype, int* _n, int &nb);
};

typedef  LBTopology* (*LBtopoFn)();

#else
typedef  void* (*LBtopoFn)();
#endif	 /* __cplusplus */

#ifdef __cplusplus
extern "C" {
#endif
void registerLBTopos();
LBtopoFn LBTopoLookup(char *);
int getTopoMaxNeighbors(void *topo);
void getTopoNeighbors(void *topo, int myid, int* na, int *n);
void printoutTopo();
#ifdef __cplusplus
}
#endif

#endif
