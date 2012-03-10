/**
 * \addtogroup CkLdb
*/
/*@{*/


#ifndef _LBTOPOLOGY_H
#define _LBTOPOLOGY_H

#define HOP_LINK_DELAY 10e-6
#define HOP_PROC_DELAY 10e-6

#ifdef __cplusplus

class LBTopology {
protected:
  int npes;
public:
  LBTopology(int p): npes(p) {}
  virtual ~LBTopology() {}
  virtual int max_neighbors() = 0;
  virtual void neighbors(int mype, int* _n, int &nb) = 0;
  //added by zshao1, these defaults mean the the topology does not support these methods
  virtual int get_dimension() { return -1;}
  virtual bool get_processor_coordinates(int processor_id, int* processor_coordinates) { return false; }
  virtual bool get_processor_id(const int* processor_coordinates, int* processor_id) { return false; }
  virtual bool coordinate_difference(const int* my_coordinates, const int* target_coordinates, int* difference) { return false;}
  virtual bool coordinate_difference(int my_processor_id, int target_processor_id, int* difference) { return false; }
  virtual int get_hop_count(int src,int dest);
  virtual int rec_hop_count(int src,int dest,int max_neigh,int count,int *visited_srcs,int min_hop_cnt);
  virtual double per_hop_delay(int last_hop);
  virtual void get_pairwise_hop_count(double **dist);
};

#define LBTOPO_MACRO(x) \
  static LBTopology * create##x(int np) { 	\
		return new x(np); 	\
  }

class LBTopo_ring: public LBTopology {
public:
  LBTopo_ring(int p): LBTopology(p) {}
  virtual int max_neighbors();
  virtual void neighbors(int mype, int* _n, int &nb);
  virtual int get_hop_count(int src,int dest);
};

class LBTopo_torus2d: public LBTopology {
private:
  int width;
  int goodcoor(int, int);
public:
  LBTopo_torus2d(int p);
  virtual int max_neighbors();
  virtual void neighbors(int mype, int* _n, int &nb);
  virtual int get_hop_count(int src,int dest);
};

class LBTopo_torus3d: public LBTopology {
private:
  int width;
  int goodcoor(int, int, int);
public:
  LBTopo_torus3d(int p);
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

typedef  LBTopology* (*LBtopoFn)(int);

#else
typedef  void* (*LBtopoFn)();
#endif	 /* __cplusplus */

#ifdef __cplusplus
extern "C" {
#endif
LBtopoFn LBTopoLookup(char *);
int getTopoMaxNeighbors(void *topo);
void getTopoNeighbors(void *topo, int myid, int* na, int *n);
void printoutTopo();
#ifdef __cplusplus
}
#endif

#endif
