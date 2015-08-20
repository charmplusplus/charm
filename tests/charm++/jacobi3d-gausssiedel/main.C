/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/** \file jacobi3d Gauss-Seidel to measure openMP, ckLoop
 *  Author: Yanhua Sun 
 *  This does a topological placement for a 3d jacobi.
 *
 *	
 *	      *****************
 *	   *		   *  *
 *   ^	*****************     *
 *   |	*		*     *
 *   |	*		*     *
 *   |	*		*     *
 *   Y	*		*     *
 *   |	*		*     *
 *   |	*		*     *
 *   |	*		*  * 
 *   ~	*****************    Z
 *	<------ X ------> 
 *
 *   X: left, right --> wrap_x
 *   Y: top, bottom --> wrap_y
 *   Z: front, back --> wrap_z
 */

#include "main.decl.h"
#include "jacobi3d.decl.h"
#include "jacobi3d.h"
#include "defines.h"
#include "TopoManager.h"

#include "CkLoopAPI.h"
CProxy_FuncCkLoop ckLoopProxy;
#ifdef JACOBI_OPENMP
#include <omp.h>
#endif

/*readonly*/ int gaussIter;  
/*readonly*/ double convergeDelta;
/*readonly*/ int threadNums;

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int arrayDimX;
/*readonly*/ int arrayDimY;
/*readonly*/ int blockDimX;
/*readonly*/ int arrayDimZ;
/*readonly*/ int blockDimY;
/*readonly*/ int blockDimZ;

// specify the number of worker chares in each dimension
/*readonly*/ int num_chare_x;
/*readonly*/ int num_chare_y;
/*readonly*/ int num_chare_z;

/*readonly*/ int globalBarrier;

static unsigned long next = 1;

double startTime;
double endTime;

class Main : public CBase_Main {
public:
    CProxy_Jacobi array;
    CProxy_TraceControl _traceControl;
    int iterations;

    void processCommandlines(int argc, char** argv)
    {
        gaussIter = 1;
        convergeDelta = 0.01;
        threadNums = 1;
        for (int i=0; i<argc; i++) {
            if (argv[i][0]=='-') {
                switch (argv[i][1]) {
                case 'X': arrayDimX = atoi(argv[++i]); break;
                case 'Y': arrayDimY = atoi(argv[++i]); break;
                case 'Z': arrayDimZ = atoi(argv[++i]); break;
                case 'x': blockDimX = atoi(argv[++i]); break;
                case 'y': blockDimY = atoi(argv[++i]); break;
                case 'z': blockDimZ = atoi(argv[++i]); break;
                case 'g': gaussIter =  atoi(argv[++i]); break;
                case 't': convergeDelta = atof(argv[++i]); break;
                case 'r': threadNums = atoi(argv[++i]); break;
                }   
            }       
        }       
        if (arrayDimX < blockDimX || arrayDimX % blockDimX != 0)
            CkAbort("array_size_X % block_size_X != 0!");
        if (arrayDimY < blockDimY || arrayDimY % blockDimY != 0)
            CkAbort("array_size_Y % block_size_Y != 0!");
        if (arrayDimZ < blockDimZ || arrayDimZ % blockDimZ != 0)
            CkAbort("array_size_Z % block_size_Z != 0!");

        num_chare_x = arrayDimX / blockDimX;
        num_chare_y = arrayDimY / blockDimY;
        num_chare_z = arrayDimZ / blockDimZ;

        // print info
        CkPrintf("\nSTENCIL COMPUTATION WITH NO BARRIERS\n");
        CkPrintf("Running Jacobi on %d processors with (%d, %d, %d) chares threshold=%f\n", CkNumPes(), num_chare_x, num_chare_y, num_chare_z, convergeDelta);
        CkPrintf("Array Dimensions: %d %d %d\n", arrayDimX, arrayDimY, arrayDimZ);
        CkPrintf("Block Dimensions: %d %d %d\n", blockDimX, blockDimY, blockDimZ);
    } 
    
    Main(CkArgMsg* m) {
        iterations = 0;
        mainProxy = thisProxy;
        processCommandlines(m->argc, m->argv);
        delete m;
#if USE_TOPOMAP
        CProxy_JacobiMap map = CProxy_JacobiMap::ckNew(num_chare_x, num_chare_y, num_chare_z);
        CkPrintf("Topology Mapping is being done ... \n");
        CkArrayOptions opts(num_chare_x, num_chare_y, num_chare_z);
        opts.setMap(map);
        array = CProxy_Jacobi::ckNew(opts);
#else
#if CKLOOP
        //map one chare to one node instead of one thread 
        CProxy_JacobiNodeMap map = CProxy_JacobiNodeMap::ckNew();
        CkArrayOptions opts(num_chare_x, num_chare_y, num_chare_z);
        opts.setMap(map);
        array = CProxy_Jacobi::ckNew(opts);
#else
        array = CProxy_Jacobi::ckNew(num_chare_x, num_chare_y, num_chare_z);
#endif
#endif

#if USE_TOPOMAP
        TopoManager tmgr;
        CkArray *jarr = array.ckLocalBranch();
        int jmap[num_chare_x][num_chare_y][num_chare_z];

        int hops=0, p;
        for(int i=0; i<num_chare_x; i++)
            for(int j=0; j<num_chare_y; j++)
                for(int k=0; k<num_chare_z; k++) {
                    jmap[i][j][k] = jarr->procNum(CkArrayIndex3D(i, j, k));
                }

        for(int i=0; i<num_chare_x; i++)
            for(int j=0; j<num_chare_y; j++)
                for(int k=0; k<num_chare_z; k++) {
                    p = jmap[i][j][k];
                    hops += tmgr.getHopsBetweenRanks(p, jmap[wrap_x(i+1)][j][k]);
                    hops += tmgr.getHopsBetweenRanks(p, jmap[wrap_x(i-1)][j][k]);
                    hops += tmgr.getHopsBetweenRanks(p, jmap[i][wrap_y(j+1)][k]);
                    hops += tmgr.getHopsBetweenRanks(p, jmap[i][wrap_y(j-1)][k]);
                    hops += tmgr.getHopsBetweenRanks(p, jmap[i][j][wrap_z(k+1)]);
                    hops += tmgr.getHopsBetweenRanks(p, jmap[i][j][wrap_z(k-1)]);
                }
        CkPrintf("Total Hops: %d\n", hops);
#endif 
        //only trace some steps instead of all steps
        _traceControl = CProxy_TraceControl::ckNew();
#if CKLOOP
        ckLoopProxy = CkLoop_Init(CkMyNodeSize());
#endif
        array.doStep();
    }

    void converge(double maxDiff)
    {
        iterations++;
        if(iterations == START_ITER)
            _traceControl.startTrace();
        if(iterations == END_ITER)
            _traceControl.endTrace();

        if(iterations == WARM_ITER)
            startTime = CmiWallTimer();                
        //if(maxDiff <= convergeDelta || iterations > 20)
        if(maxDiff <= convergeDelta )
        {
            endTime = CmiWallTimer();
            CkPrintf("Completed:\t[PEs:nodes] <char_x,y,z> \t\t chares  \t\t msgbytes \t\t iterations, \t\t  times,\t\t timeperstep(ms)\n benchmark\t[%d:%d] [ %d %d %d ] \t\t %d \t\t %d \t\t %d\t\t %.3f \t\t %.3f %d\n", CkNumPes(), CkNumNodes(), num_chare_x, num_chare_y, num_chare_z, num_chare_x *num_chare_y * num_chare_z, sizeof(double) * blockDimX*blockDimY, iterations*gaussIter, endTime - startTime, (endTime - startTime)/(iterations-WARM_ITER)/gaussIter*1000, arrayDimX*arrayDimY/CmiNumNodes()*arrayDimZ/1000000*8 );
            //array.print();
            CkExit();
        }else
        {
            if(iterations % PRINT_FREQ== 0)
                CkPrintf("iteration %d  diff=%e\n", iterations, maxDiff); 
			array.doStep();
        }
    }
};

#include "jacobi3d.cc"

#include "main.def.h"
#include "jacobi3d.def.h"
