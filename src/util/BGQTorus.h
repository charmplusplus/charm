/** \file BGQTorus.h
 *  Author: Wei Jiang 
 *  Date created: March 1st, 2012  
 *  
 */

#ifndef _BGQ_TORUS_H_
#define _BGQ_TORUS_H_

#include "converse.h"

#if CMK_BLUEGENEQ

class BGQTorusManager {
  private:
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
    BGQTorusManager();
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
