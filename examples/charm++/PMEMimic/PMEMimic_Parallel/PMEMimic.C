#include <stdio.h>
#include "PMEMimic.decl.h"
#include "ckmulticast.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ 

int     N;
int     grid_x;
int     grid_y;
int     grid_z;
int     max_iter;
int     pes_per_node;
int     grain_size;
CProxy_PMEPencil_X pme_x;
CProxy_PMEPencil_Y pme_y;
CProxy_PMEPencil_Z pme_z;

CkGroupID mCastGrpId;

class DataMsg : public CkMcastBaseMsg, public CMessage_DataMsg
{
public:
    int phrase;
    char data[2048];

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
        int node_index =  index[0]*grid_x + index[1];
        penum = node_index * pes_per_node + index[2];
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
            pes_per_node = atoi(m->argv[1]);;
            grid_x = grid_y = grid_z = atoi(m->argv[2]);
            max_iter = atoi(m->argv[3]);
        }
        grain_size = grid_x/pes_per_node;
        delete m;

    //Start the computation
      CkPrintf("Running PMEMimic on %d processors for %d elements\n",
          CkNumPes(), grid_x);
      mainProxy = thisProxy;

      CProxy_PMEMap myMap_x=CProxy_PMEMap::ckNew(0); 
      CkArrayOptions opts_x(grid_y, grid_z, pes_per_node); 
      opts_x.setMap(myMap_x);

      CProxy_PMEMap myMap_y=CProxy_PMEMap::ckNew(1); 
      CkArrayOptions opts_y(grid_x, grid_z, pes_per_node); 
      opts_y.setMap(myMap_y);

      CProxy_PMEMap myMap_z=CProxy_PMEMap::ckNew(2); 
      CkArrayOptions opts_z(grid_x, grid_y, pes_per_node); 
      opts_z.setMap(myMap_z);

      pme_x = CProxy_PMEPencil_X::ckNew(0, opts_x);
      pme_y = CProxy_PMEPencil_Y::ckNew(1, opts_y);
      pme_z = CProxy_PMEPencil_Z::ckNew(2, opts_z);

      mCastGrpId = CProxy_CkMulticastMgr::ckNew();
      CkMulticastMgr *mg = CProxy_CkMulticastMgr(mCastGrpId).ckLocalBranch();

      CProxySection_PMEPencil_X mcp_x[grid_y][grid_z];

      for (int i=0; i<grid_y; i++)
        for (int j=0; j<grid_z; j++) {
          mcp_x[i][j] = CProxySection_PMEPencil_X::ckNew(pme_x, i, i, 1, j, j, 1,  0, pes_per_node-1, 1);
          mcp_x[i][j].ckSectionDelegate(mg);
          CkCallback *cb = new CkCallback(CkIndex_PMEPencil_X::cb_client(NULL), CkArrayIndex3D(i,j,0), pme_x);
          mg->setReductionClient(mcp_x[i][j], cb);
      }

      done_pme=0;
      startTimer = CmiWallTimer();
      pme_x.start();
      
    };

    void done()
    {
        done_pme++;
        if(done_pme == grid_x*grid_x)
        {
            done_pme = 0;

            CkPrintf("PME(%d, %d, %d) on %d PEs, %d iteration, avg time:%f(ms)\n", grid_x, grid_y, grid_z, CkNumPes(), max_iter, (CmiWallTimer()-startTimer)/max_iter*1000);
            CkExit();
        }
    }
};

/*array [1D]*/
class PMEPencil_X : public CBase_PMEPencil_X
{
    int PME_index;
    int buffered_num, buffered_phrase;
    int recv_nums, iteration;
public:
  PMEPencil_X(int i)
  {
      PME_index = i;
      recv_nums = 0;
      iteration = 0;
      buffered_num = 0;
  }
  PMEPencil_X(CkMigrateMessage *m) {}

  void start()
  {
   //thisindex.x thisindex.y
    // x (yz)(x), y(x, z)(y)
    int yindex = thisIndex.x/grain_size;
    for(int x=0; x<grain_size; x++)
    {
      DataMsg *msg= new DataMsg;
      msg->phrase = 1;
      //CmiPrintf("g=%d(%d,%d,%d)==>(%d, %d,%d)\n", grain_size, thisIndex.x, thisIndex.y, thisIndex.z, x+thisIndex.z*grain_size, thisIndex.y, yindex);
      pme_y(x+thisIndex.z*grain_size, thisIndex.y, yindex ).recvTrans(msg);  
    }
  }
  void recvTrans(DataMsg *msg_recv)
  {
    int expect_num, index;
    expect_num = grid_x/pes_per_node;
    index = msg_recv->phrase;

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
        //CkPrintf("[%d, %d, %d] phrase %d, iter=%d\n", thisIndex.x, thisIndex.y, thisIndex.z, msg_recv->phrase, iteration);
        if(index == 0  ) //x (y,z) to y(x,z)
        {
            iteration++;
            if(iteration == max_iter)
            {
                mainProxy.done();
                return;
            }
            int yindex = thisIndex.x/grain_size;
            for(int x=0; x<grain_size; x++)
            {
                DataMsg *msg= new DataMsg;
                msg->phrase = msg_recv->phrase+1;
                pme_y(x+thisIndex.z*grain_size, thisIndex.y, yindex ).recvTrans(msg);  
            }
        }else if(index == 1) //y(x,z) send to z(x,y)
        {
            int zindex = thisIndex.y/grain_size;
            for(int y=0; y<grain_size; y++)
            {
                DataMsg *msg= new DataMsg;
                msg->phrase = msg_recv->phrase+1;
                pme_z(thisIndex.x, y+thisIndex.z*grain_size, zindex).recvTrans(msg); 
            }
            PME_index = 3;
            recv_nums = buffered_num;
        }else if(index == 2) //Z(x,y) send to y(x,z)
        {
            int yindex = thisIndex.y/grain_size;
            for(int z=0; z<grain_size; z++)
            {
                DataMsg *msg= new DataMsg;
                msg->phrase = msg_recv->phrase+1;
                pme_y(thisIndex.x, z+thisIndex.z*grain_size, yindex).recvTrans(msg); 
            }
        } else if(index == 3) //y(x,z) to x(y,z)
        {
            int xindex = thisIndex.x/grain_size;
            for(int y=0; y<grain_size; y++)
            {
                DataMsg *msg= new DataMsg;
                msg->phrase = 0;
                pme_x(y+grain_size*thisIndex.z, thisIndex.y, xindex).recvTrans(msg); 
            }
            PME_index = 1;
            recv_nums = buffered_num;
        }
        recv_nums = 0;
    }
    delete msg_recv;
  }
  void cb_client(CkReductionMsg *msg) {
  }
};

/*array [1D]*/
class PMEPencil_Y : public CBase_PMEPencil_Y
{
    int PME_index;
    int buffered_num, buffered_phrase;
    int recv_nums, iteration;
public:
  PMEPencil_Y(int i)
  {
      PME_index = i;
      recv_nums = 0;
      iteration = 0;
      buffered_num = 0;
  }
  PMEPencil_Y(CkMigrateMessage *m) {}

  void start()
  {
   //thisindex.x thisindex.y
    // x (yz)(x), y(x, z)(y)
    int yindex = thisIndex.x/grain_size;
    for(int x=0; x<grain_size; x++)
    {
      DataMsg *msg= new DataMsg;
      msg->phrase = 1;
      pme_y(x+thisIndex.z*grain_size, thisIndex.y, yindex ).recvTrans(msg);  
    }
  }
  void recvTrans(DataMsg *msg_recv)
  {
    int expect_num, index;
    expect_num = grid_x/pes_per_node;
    index = msg_recv->phrase;

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
        //CkPrintf("[%d, %d, %d] phrase %d, iter=%d\n", thisIndex.x, thisIndex.y, thisIndex.z, msg_recv->phrase, iteration);
        if(index == 0  ) //x (y,z) to y(x,z)
        {
            iteration++;
            if(iteration == max_iter)
            {
                mainProxy.done();
                return;
            }
            int yindex = thisIndex.x/grain_size;
            for(int x=0; x<grain_size; x++)
            {
                DataMsg *msg= new DataMsg;
                msg->phrase = msg_recv->phrase+1;
                pme_y(x+thisIndex.z*grain_size, thisIndex.y, yindex ).recvTrans(msg);  
            }
        }else if(index == 1) //y(x,z) send to z(x,y)
        {
            int zindex = thisIndex.y/grain_size;
            for(int y=0; y<grain_size; y++)
            {
                DataMsg *msg= new DataMsg;
                msg->phrase = msg_recv->phrase+1;
                pme_z(thisIndex.x, y+thisIndex.z*grain_size, zindex).recvTrans(msg); 
            }
            PME_index = 3;
            recv_nums = buffered_num;
        }else if(index == 2) //Z(x,y) send to y(x,z)
        {
            int yindex = thisIndex.y/grain_size;
            for(int z=0; z<grain_size; z++)
            {
                DataMsg *msg= new DataMsg;
                msg->phrase = msg_recv->phrase+1;
                pme_y(thisIndex.x, z+thisIndex.z*grain_size, yindex).recvTrans(msg); 
            }
        } else if(index == 3) //y(x,z) to x(y,z)
        {
            int xindex = thisIndex.x/grain_size;
            for(int y=0; y<grain_size; y++)
            {
                DataMsg *msg= new DataMsg;
                msg->phrase = 0;
                pme_x(y+grain_size*thisIndex.z, thisIndex.y, xindex).recvTrans(msg); 
            }
            PME_index = 1;
            recv_nums = buffered_num;
        }
        recv_nums = 0;
    }
    delete msg_recv;
  }
  void cb_client(CkReductionMsg *msg) {
  }
};

/*array [1D]*/
class PMEPencil_Z : public CBase_PMEPencil_Z
{
    int PME_index;
    int buffered_num, buffered_phrase;
    int recv_nums, iteration;
public:
  PMEPencil_Z(int i)
  {
      PME_index = i;
      recv_nums = 0;
      iteration = 0;
      buffered_num = 0;
  }
  PMEPencil_Z(CkMigrateMessage *m) {}

  void start()
  {
   //thisindex.x thisindex.y
    // x (yz)(x), y(x, z)(y)
    int yindex = thisIndex.x/grain_size;
    for(int x=0; x<grain_size; x++)
    {
      DataMsg *msg= new DataMsg;
      msg->phrase = 1;
      pme_y(x+thisIndex.z*grain_size, thisIndex.y, yindex ).recvTrans(msg);  
    }
  }
  void recvTrans(DataMsg *msg_recv)
  {
    int expect_num, index;
    expect_num = grid_x/pes_per_node;
    index = msg_recv->phrase;

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
        //CkPrintf("[%d, %d, %d] phrase %d, iter=%d\n", thisIndex.x, thisIndex.y, thisIndex.z, msg_recv->phrase, iteration);
        if(index == 0  ) //x (y,z) to y(x,z)
        {
            iteration++;
            if(iteration == max_iter)
            {
                mainProxy.done();
                return;
            }
            int yindex = thisIndex.x/grain_size;
            for(int x=0; x<grain_size; x++)
            {
                DataMsg *msg= new DataMsg;
                msg->phrase = msg_recv->phrase+1;
                pme_y(x+thisIndex.z*grain_size, thisIndex.y, yindex ).recvTrans(msg);  
            }
        }else if(index == 1) //y(x,z) send to z(x,y)
        {
            int zindex = thisIndex.y/grain_size;
            for(int y=0; y<grain_size; y++)
            {
                DataMsg *msg= new DataMsg;
                msg->phrase = msg_recv->phrase+1;
                pme_z(thisIndex.x, y+thisIndex.z*grain_size, zindex).recvTrans(msg); 
            }
            PME_index = 3;
            recv_nums = buffered_num;
        }else if(index == 2) //Z(x,y) send to y(x,z)
        {
            int yindex = thisIndex.y/grain_size;
            for(int z=0; z<grain_size; z++)
            {
                DataMsg *msg= new DataMsg;
                msg->phrase = msg_recv->phrase+1;
                pme_y(thisIndex.x, z+thisIndex.z*grain_size, yindex).recvTrans(msg); 
            }
        } else if(index == 3) //y(x,z) to x(y,z)
        {
            int xindex = thisIndex.x/grain_size;
            for(int y=0; y<grain_size; y++)
            {
                DataMsg *msg= new DataMsg;
                msg->phrase = 0;
                pme_x(y+grain_size*thisIndex.z, thisIndex.y, xindex).recvTrans(msg); 
            }
            PME_index = 1;
            recv_nums = buffered_num;
        }
        recv_nums = 0;
    }
    delete msg_recv;
  }
  void cb_client(CkReductionMsg *msg) {
  }
};


#include "PMEMimic.def.h"
