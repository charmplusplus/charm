#include <stdio.h>
#include "PMEMimic.decl.h"
#include "ckmulticast.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ 
#define  YH_DEBUG 0
int     N;
int     grid_x;
int     grid_y;
int     grid_z;
int     max_iter;
int     pes_per_node_type;
int     pes_per_node;
int     grain_size;
CProxy_PMEPencil_X pme_x;
CProxy_PMEPencil_Y pme_y;
CProxy_PMEPencil_Z pme_z;


class DataMsg : public CMessage_DataMsg
{
public:
    int phrase;
    char data[2048];

};

class PMEMap : public CkArrayMap 
{ 
    int offset;
public: 
    PMEMap(int off) { 
        if(pes_per_node == pes_per_node_type) offset = 0;
        else offset = off*pes_per_node_type;} 
    PMEMap(CkMigrateMessage *m){} 
    int registerArray(CkArrayIndex& numElements,CkArrayID aid) { 
        return 0; 
    } 
    int procNum(int /*arrayHdl*/,const CkArrayIndex &idx) { 
        int penum;
        int *index =  (int *)idx.data();
        int node_index =  index[0]*grid_x + index[1];
        penum = node_index * pes_per_node + index[2]+offset;
        return penum; 
    } 
}; 


/*mainchare*/
class Main : public CBase_Main
{
    double nextPhraseTimer;
    int done_pme, iteration;
public:

    Main(CkArgMsg* m)
    {
        //Process command-line arguments
 
        grid_x = grid_y = grid_z = 10;
        max_iter = 100;
        pes_per_node_type = 3;
        if(m->argc > 1)
        {
            pes_per_node = atoi(m->argv[1]);
            pes_per_node_type = atoi(m->argv[2]);
            pes_per_node_type = pes_per_node/pes_per_node_type; // 1 or 3
            grid_x = grid_y = grid_z = atoi(m->argv[3]);
            max_iter = atoi(m->argv[4]);
        }
        grain_size = grid_x/(pes_per_node_type);
        delete m;
      CkPrintf("exec pes_per_node_type(must be 3 times) grid_x iteration\n");
    //Start the computation
      CkPrintf("Running PMEMimic on %d processors for %d elements\n",
          CkNumPes(), grid_x);
      mainProxy = thisProxy;

      CProxy_PMEMap myMap_x=CProxy_PMEMap::ckNew(0); 
      CkArrayOptions opts_x(grid_y, grid_z, pes_per_node_type); 
      opts_x.setMap(myMap_x);

      CProxy_PMEMap myMap_y=CProxy_PMEMap::ckNew(1); 
      CkArrayOptions opts_y(grid_x, grid_z, pes_per_node_type); 
      opts_y.setMap(myMap_y);

      CProxy_PMEMap myMap_z=CProxy_PMEMap::ckNew(2); 
      CkArrayOptions opts_z(grid_x, grid_y, pes_per_node_type); 
      opts_z.setMap(myMap_z);

      pme_x = CProxy_PMEPencil_X::ckNew(0, opts_x);
      pme_y = CProxy_PMEPencil_Y::ckNew(1, opts_y);
      pme_z = CProxy_PMEPencil_Z::ckNew(2, opts_z);

      done_pme=0;
      nextPhraseTimer = CkWallTimer();
      pme_x.nextPhrase();
      
    };

    void done()
    {
        done_pme++;
        if(done_pme == grid_x*grid_x)
        {
            done_pme = 0;

            CkPrintf("\nPME(%d, %d, %d) on %d PEs (%d pes/node)(%d nodes)\n %d iteration, average time:%f(ms)\n", grid_x, grid_y, grid_z, CkNumPes(), pes_per_node, CkNumPes()/pes_per_node, max_iter, (CkWallTimer()-nextPhraseTimer)/max_iter*1000);
            CkExit();
        }
    }
};

/*array [3D]*/
class PMEPencil_X : public CBase_PMEPencil_X
{
    int recv_nums, iteration;
    int barrier_num;
    int phrase ;
    int expect_num;
public:
  PMEPencil_X(int i)
  {
      recv_nums = 0;
      iteration = 0;
      barrier_num = 0;
      phrase = 1;
      expect_num = grid_x/pes_per_node_type;
  }
  PMEPencil_X(CkMigrateMessage *m) {}

  void nextPhrase()
  {
   //thisindex.x thisindex.y
    // x (yz)(x), y(x, z)(y)
    int yindex = thisIndex.x/grain_size;
    for(int x=0; x<grain_size; x++)
    {
      DataMsg *msg= new DataMsg;
      msg->phrase = phrase;
#if     YH_DEBUG
      CmiPrintf("X==>y %d (%d,%d,%d)==>(%d, %d,%d)\n", grain_size, thisIndex.x, thisIndex.y, thisIndex.z, x+thisIndex.z*grain_size, thisIndex.y, yindex);
#endif
      pme_y(x+thisIndex.z*grain_size, thisIndex.y, yindex ).recvTrans(msg);  
    }
  }
  void recvTrans(DataMsg *msg_recv)
  {
    recv_nums++;
    if(recv_nums == expect_num)
    {
#if     YH_DEBUG
        CkPrintf("X [%d, %d, %d] phrase %d, iter=%d\n", thisIndex.x, thisIndex.y, thisIndex.z, msg_recv->phrase, iteration);
#endif
        phrase = msg_recv->phrase+1;
        pme_x(thisIndex.x, thisIndex.y, 0).reducePencils();
        recv_nums = 0;
    }
    delete msg_recv;
  }
  void reducePencils() {
    barrier_num++;
    if(barrier_num == pes_per_node_type)
    {
        iteration++;
        if(iteration == max_iter)
        {
            mainProxy.done();
            return;
        }
       for(int i=0; i<pes_per_node_type; i++)
       {
            pme_x(thisIndex.x, thisIndex.y, i).nextPhrase();
       }
       barrier_num = 0;
    }
  }
};

/*array [3D]*/
class PMEPencil_Y : public CBase_PMEPencil_Y
{
    int PME_index;
    int buffered_num, buffered_phrase;
    int recv_nums, iteration;
    int barrier_num;
    int phrase ;
    int expect_num;
public:
  PMEPencil_Y(int i)
  {
      PME_index = 1;
      recv_nums = 0;
      iteration = 0;
      buffered_num = 0;
      barrier_num = 0;
      phrase = 1;
      expect_num = grid_x/pes_per_node_type;
  }
  PMEPencil_Y(CkMigrateMessage *m) {}

  void nextPhrase()
  {
      if(phrase == 1) //y(x,z) send to z(x,y)
      {
          int zindex = thisIndex.y/grain_size;
            for(int y=0; y<grain_size; y++)
            {
                DataMsg *msg= new DataMsg;
                msg->phrase = phrase+1;
                pme_z(thisIndex.x, y+thisIndex.z*grain_size, zindex).recvTrans(msg); 
#if     YH_DEBUG
                CmiPrintf("y==>Z %d (%d,%d,%d)==>(%d, %d,%d)\n", grain_size, thisIndex.x, thisIndex.y, thisIndex.z, thisIndex.x, y+thisIndex.z*grain_size, zindex);
#endif
            }
            PME_index = 3; 
        } else if(phrase == 3) //y(x,z) to x(y,z)
        {
            int xindex = thisIndex.x/grain_size;
            for(int y=0; y<grain_size; y++)
            {
                DataMsg *msg= new DataMsg;
                msg->phrase = 0;
                pme_x(y+grain_size*thisIndex.z, thisIndex.y, xindex).recvTrans(msg); 
#if     YH_DEBUG
                CmiPrintf("y==>X %d (%d,%d,%d)==>(%d, %d,%d)\n", grain_size, thisIndex.x, thisIndex.y, thisIndex.z, y+grain_size*thisIndex.z, thisIndex.y, xindex);
#endif
            }
            PME_index = 1;
        }
  }
  void recvTrans(DataMsg *msg_recv)
  {
    
    if(msg_recv->phrase != PME_index)
    {
        buffered_num++;
        buffered_phrase = msg_recv->phrase;
        delete msg_recv;
        return;
    }
    recv_nums++;
    if(recv_nums == expect_num)
    {
#if     YH_DEBUG
        CkPrintf("Y [%d, %d, %d] phrase %d, iter=%d\n", thisIndex.x, thisIndex.y, thisIndex.z, msg_recv->phrase, iteration);
#endif
        phrase = msg_recv->phrase;
        pme_y(thisIndex.x, thisIndex.y, 0).reducePencils();
        recv_nums = buffered_num;
        buffered_num = 0;
    }
    delete msg_recv;
  }
  void reducePencils() {
    barrier_num++;
    if(barrier_num == pes_per_node_type)
    {
       for(int i=0; i<pes_per_node_type; i++)
       {
            pme_y(thisIndex.x, thisIndex.y, i).nextPhrase();
       }
       barrier_num = 0;
    }
  }
};

/*array [3D]*/
class PMEPencil_Z : public CBase_PMEPencil_Z
{
    int recv_nums, iteration;
    int barrier_num;
    int phrase; 
    int expect_num;
public:
  PMEPencil_Z(int i)
  {
      recv_nums = 0;
      iteration = 0;
      barrier_num = 0;
      phrase = 1;
      expect_num = grid_x/pes_per_node_type;
  }
  
  PMEPencil_Z(CkMigrateMessage *m) {}

  void nextPhrase()
  {
   //thisindex.x thisindex.y
    // Z , y(x, z)(y)
    int yindex = thisIndex.y/grain_size;
    for(int z=0; z<grain_size; z++)
    {
        DataMsg *msg= new DataMsg;
        msg->phrase = phrase+1;
        pme_y(thisIndex.x, z+thisIndex.z*grain_size, yindex).recvTrans(msg); 
#if     YH_DEBUG
        CmiPrintf("Z==>Y %d (%d,%d,%d)==>(%d, %d,%d)\n", grain_size, thisIndex.x, thisIndex.y, thisIndex.z, thisIndex.x, z+thisIndex.z*grain_size, yindex);
#endif
    }
  }
  void recvTrans(DataMsg *msg_recv)
  {
    recv_nums++;
    if(recv_nums == expect_num)
    {
#if     YH_DEBUG
        CkPrintf(" Z [%d, %d, %d] phrase %d, iter=%d\n", thisIndex.x, thisIndex.y, thisIndex.z, msg_recv->phrase, iteration);
#endif
        phrase = msg_recv->phrase;
        pme_z(thisIndex.x, thisIndex.y, 0).reducePencils();
        recv_nums = 0;
    }
    delete msg_recv;
  }
  void reducePencils() {
    barrier_num++;
    if(barrier_num == pes_per_node_type)
    {
       for(int i=0; i<pes_per_node_type; i++)
       {
            pme_z(thisIndex.x, thisIndex.y, i).nextPhrase();
       }
       barrier_num = 0;
    }
  }
};

#include "PMEMimic.def.h"
