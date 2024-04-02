/** \file stencil3d.C
 *  Author: Abhinav S Bhatele
 *  Date Created: December 28th, 2010
 *
 *  This example is written to be used with periodic measurement-based load
 *  balancers at sync. The load of some chares changes across iterations and
 *  depends on the index of the chare.
 *
 *
 *
 *        *****************
 *        *       *       *
 *   ^    *****************     *
 *   |    *       *       *
 *   |    *       *       *
 *   |    *       *       *
 *   Y    *       *       *
 *   |    *       *       *
 *   |    *       *       *
 *   |    *       *       *
 *   ~    *****************    Z
 *        <------ X ------>
 *
 *   X: left, right --> wrap_x
 *   Y: top, bottom --> wrap_y
 *   Z: front, back --> wrap_z
 */

#include "stencil3d.decl.h"
#include "TopoManager.h"
#include "papi.C"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_PAPI_grp papi_arr;
/*readonly*/ int arrayDimX;
/*readonly*/ int arrayDimY;
/*readonly*/ int arrayDimZ;
/*readonly*/ int blockDimX;
/*readonly*/ int blockDimY;
/*readonly*/ int blockDimZ;

// specify the number of worker chares in each dimension
/*readonly*/ int num_chare_x;
/*readonly*/ int num_chare_y;
/*readonly*/ int num_chare_z;

static unsigned long next = 1;

int myrand(int numpes) {
  next = next * 1103515245 + 12345;
  return((unsigned)(next/65536) % numpes);
}

// We want to wrap entries around, and because mod operator %
// sometimes misbehaves on negative values. -1 maps to the highest value.
#define wrap_x(a)    (((a)+num_chare_x)%num_chare_x)
#define wrap_y(a)    (((a)+num_chare_y)%num_chare_y)
#define wrap_z(a)    (((a)+num_chare_z)%num_chare_z)

#define index(a,b,c)    ((a)+(b)*(blockDimX+2)+(c)*(blockDimX+2)*(blockDimY+2))

#define MAX_ITER         300//60//70//100//100//100
#define LBPERIOD_ITER    155     // LB is called every LBPERIOD_ITER number of program iterations
#define CHANGELOAD       30
#define LEFT             1
#define RIGHT            2
#define TOP              3
#define BOTTOM           4
#define FRONT            5
#define BACK             6
#define DIVIDEBY7        0.14285714285714285714

#define CONFIG_ITERS 20 //Not used for now
#define CONFIG_COUNT 6
#define CONFIG_INTERVAL 10 //Same as LB Period for now
//#define WPN_LIST (int[]){24,12,8,16,12}//8}//12}
//#define WPN_LIST (int[]){24,12,8,16,8}
//#define WPN_LIST (int[]){24,12,8}
//#define WPN_LIST (int[]){24,8,12}
//#define DEBUG_ST

#define NO_REDUCTION 0
#define WORK_ITER 300//600

/** \class Main
 *
 */
class Main : public CBase_Main {
  public:
    CProxy_Stencil array;

    Main(CkArgMsg* m) {
      if ( (m->argc != 3) && (m->argc != 7) ) {
        CkPrintf("%s [array_size] [block_size]\n", m->argv[0]);
        CkPrintf("OR %s [array_size_X] [array_size_Y] [array_size_Z] [block_size_X] [block_size_Y] [block_size_Z]\n", m->argv[0]);
        CkAbort("Abort");
      }
      CkPrintf("\nNo reduction = %d, work iter = %d", NO_REDUCTION, WORK_ITER);
      // store the main proxy
      mainProxy = thisProxy;

      if(m->argc == 3) {
        arrayDimX = arrayDimY = arrayDimZ = atoi(m->argv[1]);
        blockDimX = blockDimY = blockDimZ = atoi(m->argv[2]);
      }
      else if (m->argc == 7) {
        arrayDimX = atoi(m->argv[1]);
        arrayDimY = atoi(m->argv[2]);
        arrayDimZ = atoi(m->argv[3]);
        blockDimX = atoi(m->argv[4]);
        blockDimY = atoi(m->argv[5]);
        blockDimZ = atoi(m->argv[6]);
      }

      if (arrayDimX < blockDimX || arrayDimX % blockDimX != 0)
        CkAbort("array_size_X %% block_size_X != 0!");
      if (arrayDimY < blockDimY || arrayDimY % blockDimY != 0)
        CkAbort("array_size_Y %% block_size_Y != 0!");
      if (arrayDimZ < blockDimZ || arrayDimZ % blockDimZ != 0)
        CkAbort("array_size_Z %% block_size_Z != 0!");

      num_chare_x = arrayDimX / blockDimX;
      num_chare_y = arrayDimY / blockDimY;
      num_chare_z = arrayDimZ / blockDimZ;

      // print info
      CkPrintf("\nSTENCIL COMPUTATION WITH BARRIERS\n");
      CkPrintf("Running Stencil on %d processors with (%d, %d, %d) chares\n", CkNumPes(), num_chare_x, num_chare_y, num_chare_z);
      CkPrintf("Array Dimensions: %d %d %d\n", arrayDimX, arrayDimY, arrayDimZ);
      CkPrintf("Block Dimensions: %d %d %d\n", blockDimX, blockDimY, blockDimZ);

      papi_arr = CProxy_PAPI_grp::ckNew();
      // Create new array of worker chares
      array = CProxy_Stencil::ckNew(num_chare_x, num_chare_y, num_chare_z);
      set_active_pes(CkNodeSize(CkMyNode()));
      set_active_redn_pes(CkNodeSize(CkMyNode()));
#if 1
      CkCallback cb(CkIndex_Stencil::ProcessAtSync(), array(0,0,0));
      CkStartQD(cb);
#endif
      //Start the computation
      array.doStep();
    }

    // Each worker reports back to here when it completes an iteration
    void report() {
      CkExit();
    }
};

#if 0
class PAPI_grp: public CBase_PAPI_grp {
  public:
  double end_time[MAX_ITER];
  int iter;
  PAPI_grp() {
    iter = 0;
  }
};
#endif

/** \class Stencil
 *
 */

class Stencil: public CBase_Stencil {
  Stencil_SDAG_CODE
  private:
    double startTime;
    PAPI_grp *papi;
  public:
    int iterations;
    int imsg;
    int elems, elems_a;
    int idx;

    double *temperature;
    double *new_temperature;

    // ghost arrays
    double *leftGhost;
    double *rightGhost;
    double *topGhost;
    double *bottomGhost;
    double *frontGhost;
    double *backGhost;

    // Constructor, initialize values
    Stencil() {
      usesAtSync = true;
      papi = papi_arr.ckLocalBranch();
      elems = 0;
      elems_a = 0;
      idx = 0;

      int i, j, k;
      // allocate a three dimensional array
      temperature = new double[(blockDimX+2) * (blockDimY+2) * (blockDimZ+2)];
      new_temperature = new double[(blockDimX+2) * (blockDimY+2) * (blockDimZ+2)];

      for(k=0; k<blockDimZ+2; ++k)
        for(j=0; j<blockDimY+2; ++j)
          for(i=0; i<blockDimX+2; ++i)
            temperature[index(i, j, k)] = 0.0;

      iterations = 0;
      imsg = 0;
      constrainBC();
      // start measuring time
      if (thisIndex.x == 0 && thisIndex.y == 0 && thisIndex.z == 0)
        startTime = CkWallTimer();

      // Allocate ghost arrays
      leftGhost   = new double[blockDimY*blockDimZ];
      rightGhost  = new double[blockDimY*blockDimZ];
      topGhost    = new double[blockDimX*blockDimZ];
      bottomGhost = new double[blockDimX*blockDimZ];
      frontGhost  = new double[blockDimX*blockDimY];
      backGhost   = new double[blockDimX*blockDimY];
    }

    void StartResume() {
      CkPrintf("\ninside startResum");
      thisProxy.doStep();
    }
    void ProcessAtSync(){
//      getLBMgr()->wakeupPEs();
      //set_active_redn_pes(papi->wpn);
#if 1
//      set_active_pes(papi->wpn);//CkNodeSize(CkMyNode())/2+1);
//      if(papi->config_step <= CONFIG_COUNT)
      {
//        CkCallback cb(CkIndex_Stencil::ProcessAtSync(), thisProxy(0,0,0));
//        CkStartQD(cb);
      }
#endif
#if 1//DEBUG_ST
      CkPrintf("\n----Calling AtSync");
#endif
//      thisProxy.doAtSync();
    }

    void doAtSync() {
      AtSync();
    }

    void pup(PUP::er &p)
    {
      p|startTime;
      p|iterations;
      p|imsg;
      p|elems;
      p|elems_a;
      p|idx;

      size_t size = (blockDimX+2) * (blockDimY+2) * (blockDimZ+2);
      if (p.isUnpacking()) {
        temperature     = new double[size];
        new_temperature = new double[size];
        leftGhost       = new double[blockDimY*blockDimZ];
        rightGhost      = new double[blockDimY*blockDimZ];
        topGhost        = new double[blockDimX*blockDimZ];
        bottomGhost     = new double[blockDimX*blockDimZ];
        frontGhost      = new double[blockDimX*blockDimY];
        backGhost       = new double[blockDimX*blockDimY];
      }
      p(temperature, size);
      p(new_temperature, size);
    }

    Stencil(CkMigrateMessage* m) { }

    ~Stencil() {
      delete [] temperature;
      delete [] new_temperature;
      delete [] leftGhost;
      delete [] rightGhost;
      delete [] topGhost;
      delete [] bottomGhost;
      delete [] frontGhost;
      delete [] backGhost;
    }

    // Send ghost faces to the six neighbors
    void begin_iteration(void) {
      iterations++;

      for(int k=0; k<blockDimZ; ++k)
        for(int j=0; j<blockDimY; ++j) {
          leftGhost[k*blockDimY+j] = temperature[index(1, j+1, k+1)];
          rightGhost[k*blockDimY+j] = temperature[index(blockDimX, j+1, k+1)];
        }

      for(int k=0; k<blockDimZ; ++k)
        for(int i=0; i<blockDimX; ++i) {
          topGhost[k*blockDimX+i] = temperature[index(i+1, 1, k+1)];
          bottomGhost[k*blockDimX+i] = temperature[index(i+1, blockDimY, k+1)];
        }

      for(int j=0; j<blockDimY; ++j)
        for(int i=0; i<blockDimX; ++i) {
          frontGhost[j*blockDimX+i] = temperature[index(i+1, j+1, 1)];
          backGhost[j*blockDimX+i] = temperature[index(i+1, j+1, blockDimZ)];
        }

      // Send my left face
      thisProxy(wrap_x(thisIndex.x-1), thisIndex.y, thisIndex.z)
        .receiveGhosts(iterations, RIGHT, blockDimY, blockDimZ, leftGhost);
      // Send my right face
      thisProxy(wrap_x(thisIndex.x+1), thisIndex.y, thisIndex.z)
        .receiveGhosts(iterations, LEFT, blockDimY, blockDimZ, rightGhost);
      // Send my bottom face
      thisProxy(thisIndex.x, wrap_y(thisIndex.y-1), thisIndex.z)
        .receiveGhosts(iterations, TOP, blockDimX, blockDimZ, bottomGhost);
      // Send my top face
      thisProxy(thisIndex.x, wrap_y(thisIndex.y+1), thisIndex.z)
        .receiveGhosts(iterations, BOTTOM, blockDimX, blockDimZ, topGhost);
      // Send my front face
      thisProxy(thisIndex.x, thisIndex.y, wrap_z(thisIndex.z-1))
        .receiveGhosts(iterations, BACK, blockDimX, blockDimY, frontGhost);
      // Send my back face
      thisProxy(thisIndex.x, thisIndex.y, wrap_z(thisIndex.z+1))
        .receiveGhosts(iterations, FRONT, blockDimX, blockDimY, backGhost);
    }

    void processGhosts(int dir, int height, int width, double gh[]) {
      switch(dir) {
        case LEFT:
          for(int k=0; k<width; ++k)
            for(int j=0; j<height; ++j) {
              temperature[index(0, j+1, k+1)] = gh[k*height+j];
            }
          break;
        case RIGHT:
          for(int k=0; k<width; ++k)
            for(int j=0; j<height; ++j) {
              temperature[index(blockDimX+1, j+1, k+1)] = gh[k*height+j];
            }
          break;
        case BOTTOM:
          for(int k=0; k<width; ++k)
            for(int i=0; i<height; ++i) {
              temperature[index(i+1, 0, k+1)] = gh[k*height+i];
            }
          break;
        case TOP:
          for(int k=0; k<width; ++k)
            for(int i=0; i<height; ++i) {
              temperature[index(i+1, blockDimY+1, k+1)] = gh[k*height+i];
            }
          break;
        case FRONT:
          for(int j=0; j<width; ++j)
            for(int i=0; i<height; ++i) {
              temperature[index(i+1, j+1, 0)] = gh[j*height+i];
            }
          break;
        case BACK:
          for(int j=0; j<width; ++j)
            for(int i=0; i<height; ++i) {
              temperature[index(i+1, j+1, blockDimZ+1)] = gh[j*height+i];
            }
          break;
        default:
          CkAbort("ERROR\n");
      }
    }


    void endIter() {
      elems++;
      if(elems == num_chare_x*num_chare_y*num_chare_z) {
        elems = 0;
#if DEBUG_ST
        CkPrintf("\nCalling doStep"); fflush(stdout);
#endif
        thisProxy.doStep();
      }
    }

    void endIterAtSync() {
      elems_a++;
      if(elems_a == num_chare_x*num_chare_y*num_chare_z) {
        elems_a = 0;
#if DEBUG_ST
        CkPrintf("\nCalling doStep"); fflush(stdout);
#endif
        getLBMgr()->wakeupPEs();
//        set_active_redn_pes(papi->wpn);
        thisProxy.doAtSync();
      }
    }

    void DonePgm() {
      contribute(CkCallback(CkReductionTarget(Main, report), mainProxy));
    }
    void check_and_compute() {
      compute_kernel();

      // calculate error
      // not being done right now since we are doing a fixed no. of iterations
      double *tmp;
      tmp = temperature;
      temperature = new_temperature;
      new_temperature = tmp;

      constrainBC();

      double endTime;
      if(thisIndex.x == 0 && thisIndex.y == 0 && thisIndex.z == 0) {
        endTime = CkWallTimer();
//        CkPrintf("[%d] Time per iteration: %f %f\n", iterations, (endTime - startTime), endTime);
//        fflush(stdout);
      }
      if(CkMyPe()==0) {
        if(papi->iter == iterations-1 ) {
#define ENERGY
#ifdef ENERGY
          if(iterations%10==4/*34*//*24*/) papi->start_energy(CkWallTimer());
          if(iterations%10==6/*36*//*26*/) papi->stop_energy(CkWallTimer());
#endif
          if(papi->iter > 1) {
            double iter_time = papi->end_time[papi->iter-1]-papi->end_time[papi->iter-2];
            int kth_it = iterations % CONFIG_INTERVAL;
            if(kth_it <= 8 && kth_it >5) {
              CkPrintf("\n[%d] avg_sum (%lf) += %lf", iterations, papi->avg_time, iter_time);
              papi->avg_time += iter_time;
            }
            if(kth_it == 8 && papi->config_step < CONFIG_COUNT) {
              papi->report_time(papi->wpn,papi->avg_time/3.0);
              if(papi->config_step==0) {
                papi->prev_time = papi->avg_time/3.0;
                idx++;
              } else {
                double cur_cnf_time = papi->avg_time/3.0;
                double diff = cur_cnf_time-papi->prev_time;
                int nxt = -1;
                double factor = abs(diff)/papi->prev_time;
                int skip_over = 0;
                if(factor > 0.0) skip_over = (int)floor((factor/0.15));

                if(papi->config_step==1) idx = 1;
                char* direction = "go_lower";
                if(diff < 0.2) {
                  nxt = papi->go_lower(idx, skip_over);
                } else {
                  direction = "go_higher";
                  nxt = papi->go_higher(idx, 1);
                }
                printf("\n%lf/%lf = %lf, skip_over = %d, dir=%s", abs(diff), papi->prev_time, factor, skip_over, direction);
                printf("nxt = %d", nxt);
                papi->prev_time = cur_cnf_time;
                idx = nxt;
              }
              papi->avg_time = 0.0;
              papi->wpn = WPN_LIST[idx];//papi->config_step++];
              papi->config_step++;
              if(iterations > 40 && iterations< 50) { 
                papi->wpn = 48;//47;//36;//24;//48;//papi->get_best_ppn();
                set_active_pes(papi->wpn);
              }else {
                set_active_pes(papi->wpn);
              }
              CkPrintf("\nSetting active pes = %d", papi->wpn);
            }

            CkPrintf("[%d] Time per iteration: %f %f stamp %lf\n", iterations, iter_time, papi->end_time[papi->iter-1], CkWallTimer());
            fflush(stdout);
          }
          papi->iter++;
        }
        papi->end_time[iterations-1] = CkWallTimer();
        //papi->pkg_energy[iterations-1] = papi_stop
      }

      if(iterations == MAX_ITER) {
        if(thisIndex.x==0 && thisIndex.y==0 && thisIndex.z==0) {
          for(int i=1;i<MAX_ITER;i++)
            CkPrintf("[%d] Time per iteration: %f %f\n", i+1, papi->end_time[i]-papi->end_time[i-1], papi->end_time[i]);
          CkCallback cb(CkIndex_Stencil::DonePgm(), thisProxy);
          CkStartQD(cb);
          set_active_redn_pes(CkNodeSize(CkMyNode()));
          getLBMgr()->wakeupPEs();
        }
      //  contribute(CkCallback(CkReductionTarget(Main, report), mainProxy));
      } else {

        if(thisIndex.x == 0 && thisIndex.y == 0 && thisIndex.z == 0) {
          startTime = CkWallTimer();
        }
//        if(thisIndex.x == 0 && thisIndex.y == 0 && thisIndex.z == 0 && (iterations/*+2*/) %10==0 && iterations >10 && iterations <=50) getLBMgr()->wakeupPEs();
        if((iterations/*+2*/) %10==0 && iterations >10 && iterations <=50) papi->nxt_set = 0;
        if(iterations % CONFIG_INTERVAL == 0 /*&& papi->config_step < CONFIG_COUNT*/ && iterations <=55)
        {
//          wpn = WPN_LIST[config_step++];
#if DEBUG_ST
          CkPrintf("\niterations %d mod %d = 0, config_step = %d", iterations, CONFIG_INTERVAL, papi->config_step-1);
#endif
          ;//Do nothing, wait for QD
//          AtSync();
        /*} else if(iterations % CONFIG_INTERVAL == 0 && papi->config_step++ == CONFIG_COUNT && iterations < 70){
          CkPrintf("\nBest configuration was found to be %d", papi->get_best_ppn());
          papi->wpn = papi->get_best_ppn();
          set_active_pes(papi->wpn);
          thisProxy.doAtSync(); */
          thisProxy(0,0,0).endIterAtSync();
        } else {
#if NO_REDUCTION
          thisProxy(0,0,0).endIter();
#if DEBUG_ST
          CkPrintf("\nContributed by chare %d,%d,%d",thisIndex.x, thisIndex.y, thisIndex.z);fflush(stdout);
#endif
#else
          contribute(CkCallback(CkReductionTarget(Stencil, doStep), thisProxy));
#endif
        }
      }
    }

    // Check to see if we have received all neighbor values yet
    // If all neighbor values have been received, we update our values and proceed
    void compute_kernel() {
      int index = thisIndex.x + thisIndex.y*num_chare_x + thisIndex.z*num_chare_x*num_chare_y;
      int numChares = num_chare_x * num_chare_y * num_chare_z;
      double work = WORK_ITER;//600;//300.0;//200.0;//400.0;//200.0;

#ifndef _MSC_VER
#pragma unroll
#endif
      for(int w=0; w<work; w++) {
        for(int k=1; k<blockDimZ+1; ++k)
          for(int j=1; j<blockDimY+1; ++j)
            for(int i=1; i<blockDimX+1; ++i) {
              // update my value based on the surrounding values
              new_temperature[index(i, j, k)] = (temperature[index(i-1, j, k)]
                  +  temperature[index(i+1, j, k)]
                  +  temperature[index(i, j-1, k)]
                  +  temperature[index(i, j+1, k)]
                  +  temperature[index(i, j, k-1)]
                  +  temperature[index(i, j, k+1)]
                  +  temperature[index(i, j, k)] )
                *  DIVIDEBY7;
            } // end for
      }
    }

    // Enforce some boundary conditions
    void constrainBC() {
      // Heat left, top and front faces of each chare's block
      for(int k=1; k<blockDimZ+1; ++k)
        for(int i=1; i<blockDimX+1; ++i)
          temperature[index(i, 1, k)] = 255.0;
      for(int k=1; k<blockDimZ+1; ++k)
        for(int j=1; j<blockDimY+1; ++j)
          temperature[index(1, j, k)] = 255.0;
      for(int j=1; j<blockDimY+1; ++j)
        for(int i=1; i<blockDimX+1; ++i)
          temperature[index(i, j, 1)] = 255.0;
    }

    void ResumeFromSync() {
#if DEBUG_ST
      CkPrintf("\nResume from Sync %d,%d,%d", thisIndex.x, thisIndex.y, thisIndex.z); fflush(stdout);
#endif

//      if(iterations<15)
//        thisProxy[thisIndex].doStep();
//      else
//        CkPrintf("\nResuming on PE%d at time %lf s", CkMyPe(), CkWallTimer());

        papi = papi_arr.ckLocalBranch();
        if(CkMyPe()==0 && papi->nxt_set==0) {
          CkPrintf("\n-----------Setting QD for AtSync");
          papi->nxt_set=1;
          //CkCallback cb(CkIndex_Stencil::ProcessAtSync(), thisProxy[thisIndex]);
          //CkStartQD(cb);
        }
        thisProxy(0,0,0).endIter();
    }
};

#include "stencil3d.def.h"
