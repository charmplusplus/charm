/** \file RecBipartLB.h
 *  Author: Swapnil Ghike
 *  Date Created:
 *  E-mail:ghike2@illinois.edu
 *
 */

/**
 *  \addtogroup CkLdb
 */

/*@{*/

#ifndef _RECBIPARTLB_H_
#define _RECBIPARTLB_H_

#include "CentralLB.h"
#include "RecBipartLB.decl.h"

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
    std::vector<int> q;

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

class RecBipartLB : public CentralLB {
  public:
    RecBipartLB(const CkLBOptions &opt);
    RecBipartLB(CkMigrateMessage *m) : CentralLB (m) { };

    void work(LDStats *stats);
    void pup(PUP::er &p) { CentralLB::pup(p); }

  private:
    CmiBool QueryBalanceNow(int _step);
};

#endif /* _RECBIPARTLB_H_ */

/*@}*/
