/** \file BGPTorus.h
 *  Author: Abhinav S Bhatele
 *  Date created: May 21st, 2007  
 *  
 */

#ifndef _BGP_TORUS_H_
#define _BGP_TORUS_H_

#include "converse.h"

#if CMK_BLUEGENEP

#include "dcmf.h"

class BGPTorusManager {
  private:
    DCMF_Hardware_t bgp_hwt;
    int dimX;	// dimension of the allocation in X (no. of processors)
    int dimY;	// dimension of the allocation in Y (no. of processors)
    int dimZ;	// dimension of the allocation in Z (no. of processors)
    
    int hw_NX;	// dimension of the allocation in X (no. of nodes, booted partition)
    int hw_NY;	// dimension of the allocation in Y (no. of nodes, booted partition)
    int hw_NZ;	// dimension of the allocation in Z (no. of nodes, booted partition)
    int hw_NT;  // dimension of the allocation in T (no. of processors per node, booted partition)
    
    int rn_NX;  // acutal dimension of the allocation in X for the job
    int rn_NY;  // acutal dimension of the allocation in Y for the job
    int rn_NZ;  // acutal dimension of the allocation in Z for the job
    
    int thdsPerProc; //the number of threads per process (the value of +ppn)
    int procsPerNode; //the number of processes per node
    
    int torus[4];
    char *mapping;

  public:
    BGPTorusManager() {
      int numPes = CmiNumPes();
      DCMF_Hardware(&bgp_hwt);
      procsPerNode = bgp_hwt.tSize;
      thdsPerProc = CmiMyNodeSize();
      hw_NT = procsPerNode*thdsPerProc;      

      if(CmiNumPartitions() > 1) {
        rn_NX = dimX = numPes/hw_NT;
        rn_NY = rn_NZ = 1;
        dimY = dimZ = 1;
        torus[0] = torus[1] = torus[2] = torus[3] = 0;
        return;
      }

      hw_NX = bgp_hwt.xSize;
      hw_NY = bgp_hwt.ySize;
      hw_NZ = bgp_hwt.zSize;

      //if(CmiMyPe()==0) printf("hw_NX/Y/Z/T=[%d, %d, %d, %d]\n", hw_NX, hw_NY, hw_NZ, hw_NT);

      //initialize the rn_N(*) to hw_N(*), then adjust them
      rn_NX = hw_NX;
      rn_NY = hw_NY;
      rn_NZ = hw_NZ;
      
      int max_t = 0;
      if(rn_NX * rn_NY * rn_NZ != numPes/hw_NT) {
        rn_NX = rn_NY = rn_NZ = 0;
	int rn_NT=0;
        int min_x, min_y, min_z, min_t;
        min_x = min_y = min_z = min_t = (~(-1));
        unsigned int tmp_t, tmp_x, tmp_y, tmp_z;
                     
        for(int c = 0; c < numPes/thdsPerProc; c++) {
#if (DCMF_VERSION_MAJOR >= 3)
	  DCMF_NetworkCoord_t  nc;
	  DCMF_Messager_rank2network (c, DCMF_DEFAULT_NETWORK, &nc);
	  tmp_x = nc.torus.x;
	  tmp_y = nc.torus.y;
	  tmp_z = nc.torus.z;
	  tmp_t = nc.torus.t;
#else
      	  DCMF_Messager_rank2torus(c, &tmp_x, &tmp_y, &tmp_z, &tmp_t);
#endif
 	  //if(CmiMyPe()==0){
	    //printf("Adjusting proc %d, runtime x/y/z/t=[%d, %d, %d, %d]\n", c, tmp_x, tmp_y, tmp_z, tmp_t);
          //}     	  
 
      	  if(tmp_x > rn_NX) rn_NX = tmp_x;
          if(tmp_x < min_x) min_x = tmp_x;
      	  if(tmp_y > rn_NY) rn_NY = tmp_y;
          if(tmp_y < min_y) min_y = tmp_y;
      	  if(tmp_z > rn_NZ) rn_NZ = tmp_z;
          if(tmp_z < min_z) min_z = tmp_z;
      	  if(tmp_t > rn_NT) rn_NT = tmp_t;
          if(tmp_t < min_t) min_t = tmp_t;
        }            	 
      	rn_NX = rn_NX - min_x + 1;
      	rn_NY = rn_NY - min_y + 1;
      	rn_NZ = rn_NZ - min_z + 1;

        procsPerNode = rn_NT - min_t + 1;
	hw_NT = procsPerNode * thdsPerProc;
      }
      dimX = rn_NX;
      dimY = rn_NY;
      dimZ = rn_NZ;
      dimX = dimX * hw_NT;	// assuming TXYZ

      torus[0] = bgp_hwt.xTorus;
      torus[1] = bgp_hwt.yTorus;
      torus[2] = bgp_hwt.zTorus;
      torus[3] = bgp_hwt.tTorus;
      
      mapping = getenv("BG_MAPPING");      
      
      //printf("DimN[X,Y,Z,T]=[%d,%d,%d,%d], Dim[X,Y,Z]=[%d,%d,%d]\n", rn_NX,rn_NY,rn_NZ,hw_NT,dimX, dimY, dimZ);    
    }

    ~BGPTorusManager() {
     }

    inline int getDimX() { return dimX; }
    inline int getDimY() { return dimY; }
    inline int getDimZ() { return dimZ; }

    inline int getDimNX() { return rn_NX; }
    inline int getDimNY() { return rn_NY; }
    inline int getDimNZ() { return rn_NZ; }
    inline int getDimNT() { return hw_NT; }

    inline int getProcsPerNode() { return procsPerNode; }

    inline int* isTorus() { return torus; }


    inline void rankToCoordinates(int pe, int &x, int &y, int &z) {
      x = pe % dimX;
      y = (pe % (dimX*dimY)) / dimX;
      z = pe / (dimX*dimY);
    }

    inline int coordinatesToRank(int x, int y, int z) {
      return x + (y + z*dimY) * dimX;
    }

#if 0
    inline void rankToCoordinates(int pe, int &x, int &y, int &z, int &t) {
      if(mapping==NULL || (mapping!=NULL && mapping[0]=='X')) {
        x = pe % rn_NX;
        y = (pe % (rn_NX*rn_NY)) / rn_NX;
        z = (pe % (rn_NX*rn_NY*rn_NZ)) / (rn_NX*rn_NY);
        t = pe / (rn_NX*rn_NY*rn_NZ);
      } else {
        t = pe % hw_NT;
        x = (pe % (hw_NT*rn_NX)) / hw_NT;
        y = (pe % (hw_NT*rn_NX*rn_NY)) / (hw_NT*rn_NX);
        z = pe / (hw_NT*rn_NX*rn_NY);
      }
    }

    inline int coordinatesToRank(int x, int y, int z, int t) {
      if(mapping==NULL || (mapping!=NULL && mapping[0]=='X'))
        return x + (y + (z + t*rn_NZ) * rn_NY) * rn_NX;
      else
        return t + (x + (y + z*rn_NY) * rn_NX) * hw_NT;
    }
#else
    //Make it smp-aware that each node could have different numbers of processes
    //Basically the machine is viewed as NX*NY*NZ*(#cores/node), 
    //(#cores/node) can be viewed as (#processes/node * #threads/process)
    
    inline void rankToCoordinates(int pe, int &x, int &y, int &z, int &t) {
      if(mapping==NULL || (mapping!=NULL && mapping[0]=='X')) {
        int nthRank = pe % thdsPerProc;
        //tmpPE = x+y*rn_NX+z*rn_NX*rn_NY+nthProc*rn_NX*rn_NY*rn_NZ
        int tmpPE = pe / thdsPerProc;
        x = tmpPE%rn_NX;
        y = tmpPE / rn_NX % rn_NY;
        z = tmpPE / (rn_NX*rn_NY) % rn_NZ;
        int nthProc = tmpPE / (rn_NX*rn_NY*rn_NZ);
        t = nthProc*thdsPerProc + nthRank;        
      } else {
        t = pe % (thdsPerProc*procsPerNode);
        x = pe / (thdsPerProc*procsPerNode) % rn_NX;
        y = pe / (thdsPerProc*procsPerNode*rn_NX) % (rn_NY);
        z = pe / (thdsPerProc*procsPerNode*rn_NX*rn_NY);
      }
    }

    inline int coordinatesToRank(int x, int y, int z, int t) {
      if(mapping==NULL || (mapping!=NULL && mapping[0]=='X')){
        int pe;
        int nthProc = t/thdsPerProc;
        int nthRank = t%thdsPerProc;
        
        pe = (x+y*rn_NX+z*rn_NX*rn_NY)*thdsPerProc;
        
        pe += nthProc*rn_NX*rn_NY*rn_NZ*thdsPerProc + nthRank;
        
        return pe;
      }      
      else
        return t + (x + (y + z*rn_NY) * rn_NX) * thdsPerProc * procsPerNode;
    }    
     
#endif        

	inline int getTotalPhyNodes(){
		return rn_NX * rn_NY * rn_NZ;
	}
	inline int getMyPhyNodeID(int pe){
		int x,y,z,t;
		rankToCoordinates(pe, x, y, z, t);
		return x+y*rn_NX+z*rn_NX*rn_NY;
	}
	
};

#endif // CMK_BLUEGENEP
#endif //_BGP_TORUS_H_
