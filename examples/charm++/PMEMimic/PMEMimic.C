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
            pes_per_node = atoi(m->argv[1]);;
            grid_x = grid_y = grid_z = atoi(m->argv[2]);
            max_iter = atoi(m->argv[3]);
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

      pme_x = CProxy_PMEPencil_X::ckNew(0, opts_x);
      pme_y = CProxy_PMEPencil_Y::ckNew(1, opts_y);
      pme_z = CProxy_PMEPencil_Z::ckNew(2, opts_z);

      done_pme=0;
      startTimer = CkWallTimer();
      pme_x.start();
      
    };

    void done()
    {
        done_pme++;
        if(done_pme == grid_x*grid_x)
        {
            done_pme = 0;

            CkPrintf("PME(%d, %d, %d) on %d PEs, %d iteration, avg time:%f(ms)\n", grid_x, grid_y, grid_z, CkNumPes(), max_iter, (CkWallTimer()-startTimer)/max_iter*1000);
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
    expect_num = grid_x;
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
        //CkPrintf("[%d, %d] phrase %d, iter=%d\n", thisIndex.x, thisIndex.y, msg_recv->phrase, iteration);
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
                msg->phrase = msg_recv->phrase+1;
                pme_y(x, thisIndex.y).recvTrans(msg);  
            }
        }else if(index == 1) //y(x,z) send to z(x,y)
        {
            for(int y=0; y<grid_y; y++)
            {
                DataMsg *msg= new DataMsg;
                msg->phrase = msg_recv->phrase+1;
                pme_z(thisIndex.x, y).recvTrans(msg); 
            }
            PME_index = 3;
            recv_nums = buffered_num;
        }else if(index == 2) //Z(x,y) send to y(x,z)
        {
            for(int z=0; z<grid_z; z++)
            {
                DataMsg *msg= new DataMsg;
                msg->phrase = msg_recv->phrase+1;
                pme_y(thisIndex.x, z).recvTrans(msg); 
            }
        } else if(index == 3) //y(x,z) to x(y,z)
        {
            for(int y=0; y<grid_y; y++)
            {
                DataMsg *msg= new DataMsg;
                msg->phrase = 0;
                pme_x(y, thisIndex.y).recvTrans(msg); 
            }
            PME_index = 1;
            recv_nums = buffered_num;
        }
        recv_nums = 0;
    }
    delete msg_recv;
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
    expect_num = grid_x;
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
        //CkPrintf("[%d, %d] phrase %d, iter=%d\n", thisIndex.x, thisIndex.y, msg_recv->phrase, iteration);
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
                msg->phrase = msg_recv->phrase+1;
                pme_y(x, thisIndex.y).recvTrans(msg);  
            }
        }else if(index == 1) //y(x,z) send to z(x,y)
        {
            for(int y=0; y<grid_y; y++)
            {
                DataMsg *msg= new DataMsg;
                msg->phrase = msg_recv->phrase+1;
                pme_z(thisIndex.x, y).recvTrans(msg); 
            }
            PME_index = 3;
            recv_nums = buffered_num;
        }else if(index == 2) //Z(x,y) send to y(x,z)
        {
            for(int z=0; z<grid_z; z++)
            {
                DataMsg *msg= new DataMsg;
                msg->phrase = msg_recv->phrase+1;
                pme_y(thisIndex.x, z).recvTrans(msg); 
            }
        } else if(index == 3) //y(x,z) to x(y,z)
        {
            for(int y=0; y<grid_y; y++)
            {
                DataMsg *msg= new DataMsg;
                msg->phrase = 0;
                pme_x(y, thisIndex.y).recvTrans(msg); 
            }
            PME_index = 1;
            recv_nums = buffered_num;
        }
        recv_nums = 0;
    }
    delete msg_recv;
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
    expect_num = grid_x;
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
        //CkPrintf("[%d, %d] phrase %d, iter=%d\n", thisIndex.x, thisIndex.y, msg_recv->phrase, iteration);
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
                msg->phrase = msg_recv->phrase+1;
                pme_y(x, thisIndex.y).recvTrans(msg);  
            }
        }else if(index == 1) //y(x,z) send to z(x,y)
        {
            for(int y=0; y<grid_y; y++)
            {
                DataMsg *msg= new DataMsg;
                msg->phrase = msg_recv->phrase+1;
                pme_z(thisIndex.x, y).recvTrans(msg); 
            }
            PME_index = 3;
            recv_nums = buffered_num;
        }else if(index == 2) //Z(x,y) send to y(x,z)
        {
            for(int z=0; z<grid_z; z++)
            {
                DataMsg *msg= new DataMsg;
                msg->phrase = msg_recv->phrase+1;
                pme_y(thisIndex.x, z).recvTrans(msg); 
            }
        } else if(index == 3) //y(x,z) to x(y,z)
        {
            for(int y=0; y<grid_y; y++)
            {
                DataMsg *msg= new DataMsg;
                msg->phrase = 0;
                pme_x(y, thisIndex.y).recvTrans(msg); 
            }
            PME_index = 1;
            recv_nums = buffered_num;
        }
        recv_nums = 0;
    }
    delete msg_recv;
  }
};



#include "PMEMimic.def.h"
