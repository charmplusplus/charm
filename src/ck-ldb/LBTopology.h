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

#define LB_RING          1
#define LB_MESH2D        2
#define LB_MESH3D        3
#define LB_TORUS2D       4
#define LB_TORUS3D       5

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
    if (_lb_debug && CkMyPe()==0) CkPrintf("LB> Topology %s\n", #x);	\
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

typedef  LBTopology* (*LBtopoFn)();

extern void registerLBTopos();
extern LBtopoFn LBTopoLookup(char *);

#endif
