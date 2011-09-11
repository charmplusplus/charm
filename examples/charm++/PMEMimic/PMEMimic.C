#include <stdio.h>
#include "PMEMimic.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ 

int     N;
int     grid_x;
int     grid_y;
int     grid_z;
int     max_iter;
int     pes_per_node;

CProxy_PMEPencil pme_x;
CProxy_PMEPencil pme_y;
CProxy_PMEPencil pme_z;

class DataMsg : public CMessage_DataMsg
{
public:
    int phrase;
    char data[2048];

    DataMsg() {}

};

class PMEMap : public CkArrayMap 
{ 
    int offset;
public: 
    PMEMap(int off) { offset = off;} 
    PMEMap(CkMigrateMessage *m){} 
    int registerArray(CkArrayIndex& numElements,CkArrayID aid) { 
        return 0; 
    } 
    int procNum(int /*arrayHdl*/,const CkArrayIndex &idx) { 
        int penum;
        int *index =  (int *)idx.data();
        int obj_index =  index[0]*grid_x + index[1];
        penum = obj_index * pes_per_node + offset;
        return penum; 
    } 
}; 


/*mainchare*/
class Main : public CBase_Main
{
    double startTimer;
    int done_pme, iteration;
public:

    Main(CkArgMsg* m)
    {
        //Process command-line arguments
 
        grid_x = grid_y = grid_z = 10;
        max_iter = 100;
        pes_per_node = 3;
        if(m->argc > 1)
        {
            pes_per_node = 3;
            grid_x = grid_y = grid_z = atoi(m->argv[1]);
            max_iter = atoi(m->argv[2]);
        }
        delete m;

    //Start the computation
      CkPrintf("Running PMEMimic on %d processors for %d elements\n",
          CkNumPes(), grid_x);
      mainProxy = thisProxy;

      CProxy_PMEMap myMap_x=CProxy_PMEMap::ckNew(0); 
      CkArrayOptions opts_x(grid_y, grid_z); 
      opts_x.setMap(myMap_x);

      CProxy_PMEMap myMap_y=CProxy_PMEMap::ckNew(1); 
      CkArrayOptions opts_y(grid_x, grid_z); 
      opts_y.setMap(myMap_y);

      CProxy_PMEMap myMap_z=CProxy_PMEMap::ckNew(2); 
      CkArrayOptions opts_z(grid_x, grid_y); 
      opts_z.setMap(myMap_z);

      pme_x = CProxy_PMEPencil::ckNew(opts_x);
      pme_y = CProxy_PMEPencil::ckNew(opts_y);
      pme_z = CProxy_PMEPencil::ckNew(opts_z);

      done_pme=0;
      startTimer = CmiWallTimer();
      pme_x.start();
    };

    void done()
    {
        done_pme++;
        if(done_pme == grid_x)
        {
            done_pme = 0;

            CkPrintf("PME(%d, %d, %d) on %d PEs, %d iteration, avg time:%f\n", grid_x, grid_y, grid_z, CkNumPes(), max_iter, CmiWallTimer()-startTimer);
            CkExit();
        }
    }
};

/*array [1D]*/
class PMEPencil : public CBase_PMEPencil
{
    int recv_nums, iteration;
public:
  PMEPencil()
  {
    recv_nums = 0;
    iteration = 0;
  }
  PMEPencil(CkMigrateMessage *m) {}

  void start()
  {
   //thisindex.x thisindex.y
    // x (yz), y(x, z)
    for(int x=0; x<grid_x; x++)
    {
      DataMsg *msg= new DataMsg;
      msg->phrase = 1;
      pme_y(x, thisIndex.y).recvTrans(msg);  
    }
  }
  void recvTrans(DataMsg *msg_recv)
  {
    int expect_num, index;
    recv_nums++;
    expect_num = grid_x;
    index = msg_recv->phrase;

    if(recv_nums == expect_num)
    {
        if(index == 0  ) //x (y,z) to y(x,z)
        {
            iteration++;
            if(iteration == max_iter)
            {
                mainProxy.done();
                return;
            }
            for(int x=0; x<grid_x; x++)
            {
                DataMsg *msg= new DataMsg;
                msg->phrase = 1;
                pme_y(x, thisIndex.y).recvTrans(msg);  
            }
            CkPrintf("x==>y\n");
        }else if(index == 1) //y(x,z) send to z(x,y)
        {
            for(int y=0; y<grid_y; y++)
            {
                DataMsg *msg= new DataMsg;
                msg->phrase = msg->phrase+1;
                pme_z(thisIndex.x, y).recvTrans(msg); 
            }
            CkPrintf("y==>z\n");
        }else if(index == 2) //Z(x,y) send to y(x,z)
        {
            for(int z=0; z<grid_z; z++)
            {
                DataMsg *msg= new DataMsg;
                msg->phrase = msg->phrase+1;
                pme_z(thisIndex.x, z).recvTrans(msg); 
            }
            CkPrintf("z==>y\n");
        } else if(index == 3) //y(x,z) to x(y,z)
        {
            for(int y=0; y<grid_y; y++)
            {
                DataMsg *msg= new DataMsg;
                msg->phrase = 0;
                pme_z(y, thisIndex.y).recvTrans(msg); 
            }
            CkPrintf("y==>x\n");

        }
        recv_nums = 0;
    }
    delete msg_recv;
  }
};


#include "PMEMimic.def.h"
