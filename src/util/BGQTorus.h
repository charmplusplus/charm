/** \file BGQTorus.h
 *  Author: Wei Jiang 
 *  Date created: March 1st, 2012  
 *  
 */

#ifndef _BGQ_TORUS_H_
#define _BGQ_TORUS_H_

#include "converse.h"

#if CMK_BLUEGENEQ

#include "spi/include/kernel/process.h"
#include "spi/include/kernel/location.h"
#include <firmware/include/personality.h>

class BGQTorusManager {
  private:
    Personality_t pers;
    int dimA;	// dimension of the allocation in A (no. of processors)
    int dimB;	// dimension of the allocation in B (no. of processors)
    int dimC;	// dimension of the allocation in C (no. of processors)
    int dimD;   // dimension of the allocation in D (no. of processors)
    int dimE;   // dimension of the allocation in E (no. of processors)
    
    int hw_NA;	// dimension of the allocation in A (no. of nodes, booted partition)
    int hw_NB;	// dimension of the allocation in B (no. of nodes, booted partition)
    int hw_NC;	// dimension of the allocation in C (no. of nodes, booted partition)
    int hw_ND;  // dimension of the allocation in D (no. of nodes, booted partition)  
    int hw_NE;  // dimension of the allocation in E (no. of nodes, booted partition)
    int hw_NT; // dimension of processors per node.   
  
    int rn_NA;  // acutal dimension of the allocation in A for the job
    int rn_NB;  // acutal dimension of the allocation in B for the job
    int rn_NC;  // acutal dimension of the allocation in C for the job
    int rn_ND;  // acutal dimension of the allocation in D for the job
    int rn_NE;  // acutal dimension of the allocation in E for the job
    
    int thdsPerProc; //the number of threads per process (the value of +ppn)
    int procsPerNode; //the number of processes per node
    int torus[5];    
    int order[6], dims[6];
    char *mapping; //temporarily only conside default mapping ABCDET

  public:
    BGQTorusManager() {
      order[0] = 5;
      order[1] = 4;
      order[2] = 3;
      order[3] = 2;
      order[4] = 1;
      order[5] = 0;
      
      int numPes = CmiNumPes();
      procsPerNode = Kernel_ProcessCount();
      thdsPerProc = CmiMyNodeSize();
      hw_NT = procsPerNode*thdsPerProc;      

      if(CmiNumPartitions() > 1) {
        dimA = rn_NA = numPes/hw_NT;
        dimB = dimC = dimD = dimE = 1;
        rn_NB = rn_NC = rn_ND = rn_NE = 1;
        torus[0] = torus[1] = torus[2] = torus[3] = torus[4] = 0;
        dims[0] = rn_NA;
        dims[1] = rn_NB;
        dims[2] = rn_NC;
        dims[3] = rn_ND;
        dims[4] = rn_NE;
        dims[5] = hw_NT; 
        return;
      }

      Kernel_GetPersonality(&pers, sizeof(pers)); 

      hw_NA = pers.Network_Config.Anodes;
      hw_NB = pers.Network_Config.Bnodes;
      hw_NC = pers.Network_Config.Cnodes;
      hw_ND = pers.Network_Config.Dnodes;
      hw_NE = pers.Network_Config.Enodes;

      mapping = getenv("RANK_ORDER");
      if(mapping != NULL) {
        sscanf(mapping,"%d %d %d %d %d %d",&order[5],&order[4],&order[3],&order[2],&order[1],&order[0]);
      }
      //printf("Mapping %d %d %d %d %d %d\n",order[0],order[1],order[2],order[3],order[4],order[5]);

      rn_NA = hw_NA;
      rn_NB = hw_NB;
      rn_NC = hw_NC;
      rn_ND = hw_ND;
      rn_NE = hw_NE;

      int max_t = 0;
      if(rn_NA * rn_NB * rn_NC * rn_ND * rn_NE != numPes/hw_NT) {
        rn_NA = rn_NB = rn_NC = rn_ND =rn_NE =0;
        int rn_NT=0;
        int min_a, min_b, min_c, min_d, min_e, min_t;
        min_a = min_b = min_c = min_d = min_e = min_t = (~(-1));
        int tmp_t, tmp_a, tmp_b, tmp_c, tmp_d, tmp_e;
        uint64_t numentries;
        BG_CoordinateMapping_t *coord;

        int nranks=numPes/thdsPerProc;
        coord = (BG_CoordinateMapping_t *) malloc(sizeof(BG_CoordinateMapping_t)*nranks);
        Kernel_RanksToCoords(sizeof(BG_CoordinateMapping_t)*nranks, coord, &numentries);

        for(int c = 0; c < nranks; c++) {
          tmp_a = coord[c].a;
          tmp_b = coord[c].b;
          tmp_c = coord[c].c;
          tmp_d = coord[c].d;
          tmp_e = coord[c].e;
          tmp_t = coord[c].t;
       

          if(tmp_a > rn_NA) rn_NA = tmp_a;
          if(tmp_a < min_a) min_a = tmp_a;
          if(tmp_b > rn_NB) rn_NB = tmp_b;
          if(tmp_b < min_b) min_b = tmp_b;
          if(tmp_c > rn_NC) rn_NC = tmp_c;
          if(tmp_c < min_c) min_c = tmp_c;
          if(tmp_d > rn_ND) rn_ND = tmp_d;
          if(tmp_d < min_d) min_d = tmp_d;
          if(tmp_e > rn_NE) rn_NE = tmp_e;
          if(tmp_e < min_e) min_e = tmp_e;
          if(tmp_t > rn_NT) rn_NT = tmp_t;
          if(tmp_t < min_t) min_t = tmp_t;
          }
          rn_NA = rn_NA - min_a + 1;
          rn_NB = rn_NB - min_b + 1;
          rn_NC = rn_NC - min_c + 1;
          rn_ND = rn_ND - min_d + 1;
          rn_NE = rn_NE - min_e + 1; 
          procsPerNode = rn_NT - min_t + 1;
          hw_NT = procsPerNode * thdsPerProc;
          }
          
      dimA = rn_NA;
      dimB = rn_NB;
      dimC = rn_NC;
      dimD = rn_ND;
      dimE = rn_NE;  
      dimA = dimA * hw_NT;	// assuming TABCDE
    
      dims[0] = rn_NA;
      dims[1] = rn_NB;
      dims[2] = rn_NC;
      dims[3] = rn_ND;
      dims[4] = rn_NE;
      dims[5] = hw_NT; 
      
      torus[0] = ((rn_NA % 4) == 0)? true:false;
      torus[1] = ((rn_NB % 4) == 0)? true:false;
      torus[2] = ((rn_NC % 4) == 0)? true:false;
      torus[3] = ((rn_ND % 4) == 0)? true:false;
      torus[4] = true;

    }

    ~BGQTorusManager() {
     }

    inline int getDimX() { return dimA*dimB; }
    inline int getDimY() { return dimC*dimD; }
    inline int getDimZ() { return dimE; }

    inline int getDimNX() { return rn_NA*rn_NB; }
    inline int getDimNY() { return rn_NC*rn_ND; }
    inline int getDimNZ() { return rn_NE; }
    inline int getDimNT() { return hw_NT; }

    inline int getDimNA() { return rn_NA; }
    inline int getDimNB() { return rn_NB; }
    inline int getDimNC() { return rn_NC; }
    inline int getDimND() { return rn_ND; }
    inline int getDimNE() { return rn_NE; }

    inline int getProcsPerNode() { return procsPerNode; }
    inline int* isTorus() { return torus; }

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
    
    inline void rankToCoordinates(int pe, int &x, int &y, int &z, int &t) { //5D Torus mapping to 3D logical network, don't know if it is useful!
        t = pe % (thdsPerProc*procsPerNode);
        int e = pe / (thdsPerProc*procsPerNode) % rn_NE;
        int d = pe / (thdsPerProc*procsPerNode*rn_NE) % (rn_ND);
        int c = pe / (thdsPerProc*procsPerNode*rn_NE*rn_ND) % (rn_NC);
        int b = pe / (thdsPerProc*procsPerNode*rn_NE*rn_ND*rn_NC) % (rn_NB);
        int a = pe / (thdsPerProc*procsPerNode*rn_NE*rn_ND*rn_NC*rn_NB);

        z = e;
        y = (c + 1) * rn_ND - 1 - (((c % 2)==0)?(rn_ND - d - 1) : d);
        x = (a + 1) * rn_NB - 1 - (((a % 2)==0)?(rn_NB - b - 1) : b);
    }

    inline void rankToCoordinates(int pe, int &a, int &b, int &c, int &d, int &e, int &t) {
        int tempdims[6];

        tempdims[order[0]] = pe % dims[order[0]];
        tempdims[order[1]] = (pe / dims[order[0]]) % dims[order[1]];
        tempdims[order[2]] = (pe / (dims[order[0]]*dims[order[1]])) % dims[order[2]];
        tempdims[order[3]] = (pe / (dims[order[0]]*dims[order[1]]*dims[order[2]])) % dims[order[3]];
        tempdims[order[4]] = (pe / (dims[order[0]]*dims[order[1]]*dims[order[2]]*dims[order[3]])) % dims[order[4]];
        tempdims[order[5]] = (pe / (dims[order[0]]*dims[order[1]]*dims[order[2]]*dims[order[3]]*dims[order[4]])) % dims[order[5]];
        
        a = tempdims[0];
        b = tempdims[1];
        c = tempdims[2];
        d = tempdims[3];
        e = tempdims[4];
        t = tempdims[5];

        /*t = pe % (thdsPerProc*procsPerNode);
        e = pe / (thdsPerProc*procsPerNode) % rn_NE;
        d = pe / (thdsPerProc*procsPerNode*rn_NE) % (rn_ND);
        c = pe / (thdsPerProc*procsPerNode*rn_NE*rn_ND) % (rn_NC);
        b = pe / (thdsPerProc*procsPerNode*rn_NE*rn_ND*rn_NC) % (rn_NB);
        a = pe / (thdsPerProc*procsPerNode*rn_NE*rn_ND*rn_NC*rn_NB);
        */
    }
   
    inline int coordinatesToRank(int x, int y, int z, int t) {  //3D logic mapping to 5D torus, don't know if it is useful!
        int pe;
        int a = x/rn_NB;
        int b = ((a % 2)==0)?(x % rn_NB) : (rn_NB - (x % rn_NB) - 1);
        int c = y/rn_ND;
        int d = ((c % 2)==0)?(y % rn_ND) : (rn_ND - (y % rn_ND) - 1);
        int e = z;

        int a_mult = rn_NB * rn_NC * rn_ND * rn_NE;
        int b_mult = rn_NC * rn_ND * rn_NE;
        int c_mult = rn_ND * rn_NE;
        int d_mult = rn_NE;
        
        pe = (a * a_mult + b * b_mult + c * c_mult + d * d_mult + e) * thdsPerProc * procsPerNode + t;
        
        return pe;
    }    

    inline int coordinatesToRank(int a, int b, int c, int d, int e, int t) {
        int pe;
        int tempdims[6];
        tempdims[0] = a;
        tempdims[1] = b;
        tempdims[2] = c;
        tempdims[3] = d;
        tempdims[4] = e;
        tempdims[5] = t;

        pe = 0;

        pe += tempdims[order[0]];
        pe += (tempdims[order[1]]*dims[order[0]]);
        pe += (tempdims[order[2]]*dims[order[0]]*dims[order[1]]);
        pe += (tempdims[order[3]]*dims[order[0]]*dims[order[1]]*dims[order[2]]);
        pe += (tempdims[order[4]]*dims[order[0]]*dims[order[1]]*dims[order[2]]*dims[order[3]]);
        pe += (tempdims[order[5]]*dims[order[0]]*dims[order[1]]*dims[order[2]]*dims[order[3]]*dims[order[4]]);

        /*
        int a_mult = rn_NB * rn_NC * rn_ND * rn_NE;
        int b_mult = rn_NC * rn_ND * rn_NE;
        int c_mult = rn_ND * rn_NE;
        int d_mult = rn_NE;

        pe = (a * a_mult + b * b_mult + c * c_mult + d * d_mult + e) * thdsPerProc * procsPerNode + t;
        */

        return pe;
    }
     
#endif        

	inline int getTotalPhyNodes(){
		return rn_NA * rn_NB * rn_NC * rn_ND * rn_NE;
	}
	inline int getMyPhyNodeID(int pe){
		int x,y,z,t;
		rankToCoordinates(pe, x, y, z, t);
                int a = x/rn_NB;
                int b = ((a % 2)==0)?(x % rn_NB) : (rn_NB - (x % rn_NB) - 1);
                int c = y/rn_ND;
                int d = ((c % 2)==0)?(y % rn_ND) : (rn_ND - (y % rn_ND) - 1);
                int e = z;                
		return a * rn_NB * rn_NC * rn_ND * rn_NE + b * rn_NC * rn_ND * rn_NE + c * rn_ND * rn_NE + d * rn_NE + e;
	}
	
};

#endif // CMK_BLUEGENEQ
#endif //_BGQ_TORUS_H_
