#include "Array1D.h"
#include "RRMap.h"
#include "jacobi.top.h"

enum {cell_sz = 66};
enum {array_dim = 8};
enum {run_done = 500};

int bocnum;

#define VAL(a,i,j) (*((a) + (i) * cell_sz + (j)))
#define CELL(i,j) ((i)*array_dim + (j))


class RedMsg : public comm_object
{
  public :
  	double val;
};

class Cell;

class Reduce : public groupmember
{
  private :
	int next;
	int count;
	int numExpected;
	double max;
	Cell *ids[50];
  public :
	Reduce(RedMsg *);
	void Deposit(double val, Cell *id, int expected);
	void Recv_bcast(RedMsg *m);
	void Recv_data(RedMsg *m);
};


class NeighborMsg : public ArrayMessage
{
public:
  int which_neighbor;
  double data[cell_sz];
};

class GoMsg : public ArrayMessage
{
public:
  int go;
};

class main : public chare_object
{
public:
  main(int argc, char *argv[])
  {
    int arrayGroup;
    RedMsg *msg=(RedMsg *) new(MsgIndex(RedMsg)) RedMsg;
    bocnum=new_group(Reduce, RedMsg, msg);
    ReadInit(bocnum);
    arrayGroup =
      Array1D::CreateArray(array_dim*array_dim,
		   ChareIndex(RRMap),
		   ConstructorIndex(RRMap,ArrayMapCreateMessage),
		   ChareIndex(Cell),
		   ConstructorIndex(Cell,ArrayElementCreateMessage),
		   ConstructorIndex(Cell,ArrayElementMigrateMessage));

  }
};

class Cell : public ArrayElement
{
public:
  Cell(ArrayElementCreateMessage *msg) : ArrayElement(msg)
  {
    my_x = thisIndex % array_dim;
    my_y = thisIndex / array_dim;
    num_neighbors = 0;
    if (my_x > 0)
      num_neighbors++;
    if (my_y < array_dim-1)
      num_neighbors++;
    if (my_x < array_dim-1)
      num_neighbors++;
    if (my_y > 0)
      num_neighbors++;

    int i,j;
    for(i=0; i < cell_sz; i++)
      for(j=0; j < cell_sz; j++)
	data1[i][j] = 0;

    if (my_x==0)
    {
      for(j=0; j < cell_sz; j++)
	data1[j][0] = 100;
    }

    old_data = &(data1[0][0]);
    new_data = &(data2[0][0]);
    neighbors_reported = 0;
    run_until = 0;

    go_nogo();

    finishConstruction();
  }

  Cell(ArrayElementMigrateMessage *msg) : ArrayElement(msg)
  {
    finishMigration();
  }

  void neighbor_data(NeighborMsg *msg)
  {
    int neighbor_side = (msg->which_neighbor + 2) % 4;
    int i;

    if (neighbor_side == 0)
      for(i=1;i < cell_sz-1; i++)
	VAL(old_data,0,i) = msg->data[i];
    else if (neighbor_side == 1)
      for(i=1;i < cell_sz-1; i++)
	VAL(old_data,i,cell_sz-1) = msg->data[i];
    else if (neighbor_side == 2)
      for(i=1;i < cell_sz-1; i++)
	VAL(old_data,cell_sz-1,i) = msg->data[i];
    else 
      for(i=1;i < cell_sz-1; i++)
	VAL(old_data,i,0) = msg->data[i];
    neighbors_reported++;
    /*
    CPrintf(
      "Index %d,%d received from neighbor %d, reported=%d expected = %d\n",
      my_x,my_y,neighbor_side,neighbors_reported,num_neighbors);
      */

    delete msg;

    if (neighbors_reported == num_neighbors)
    {
      int i,j;
      for(i=1; i < cell_sz-1; i++)
	for(j=1; j < cell_sz-1; j++)
	  VAL(new_data,i,j) = 0.25*(VAL(old_data,i,j+1) 
	    + VAL(old_data,i+1,j) + VAL(old_data,i,j-1) 
	    + VAL(old_data,i-1,j));

      double *tmp_data = old_data;
      old_data = new_data;
      new_data = tmp_data;
      neighbors_reported = 0;
    //  go_nogo();
    int numClients=thisArray->num_local();
    (CLocalBranch(Reduce,ReadValue(bocnum)))->Deposit((double)0.0, (Cell *)this, numClients);
    }  
  }

  void go_nogo()
  {
    run_until++;
    //CPrintf("go_nogo(): Index %d,%d count %d\n",my_x,my_y,run_until);
    if (run_until > run_done)
    {
      CharmExit();
      return;
    }

    if (my_x > 0)
    {
      NeighborMsg *msg = new(MsgIndex(NeighborMsg)) NeighborMsg;

      msg->which_neighbor = 0;
      for(int i=1; i < cell_sz-1; i++)
	msg->data[i-1] = VAL(old_data,0,i);
      int send_to = CELL(my_x-1,my_y);
      //CPrintf("Index %d,%d sending to %d\n",my_x,my_y,send_to);
      thisArray->send(msg,send_to,EntryIndex(Cell,neighbor_data,NeighborMsg));
    }

    if (my_y < array_dim-1)
    {
      NeighborMsg *msg = new(MsgIndex(NeighborMsg)) NeighborMsg;

      msg->which_neighbor = 1;
      for(int i=1; i < cell_sz-1; i++)
	msg->data[i-1] = VAL(old_data,i,cell_sz-1);
      int send_to = CELL(my_x,my_y+1);
      //CPrintf("Index %d,%d sending to %d\n",my_x,my_y,send_to);
      thisArray->send(msg,send_to,EntryIndex(Cell,neighbor_data,NeighborMsg));
    }

    if (my_x < array_dim-1)
    {
      NeighborMsg *msg = new(MsgIndex(NeighborMsg)) NeighborMsg;

      msg->which_neighbor = 2;
      for(int i=1; i < cell_sz-1; i++)
	msg->data[i-1] = VAL(old_data,cell_sz-1,i);
      int send_to = CELL(my_x+1,my_y);
      //CPrintf("Index %d,%d sending to %d\n",my_x,my_y,send_to);
      thisArray->send(msg,send_to,EntryIndex(Cell,neighbor_data,NeighborMsg));
    }

    if (my_y > 0)
    {
      NeighborMsg *msg = new(MsgIndex(NeighborMsg)) NeighborMsg;

      msg->which_neighbor = 3;
      for(int i=1; i < cell_sz-1; i++)
	msg->data[i-1] = VAL(old_data,i,0);
      int send_to = CELL(my_x,my_y-1);
      //CPrintf("Index %d,%d sending to %d\n",my_x,my_y,send_to);
      thisArray->send(msg,send_to,EntryIndex(Cell,neighbor_data,NeighborMsg));
    }
  }

private:
  double data1[cell_sz][cell_sz];
  double data2[cell_sz][cell_sz];
  double *old_data;
  double *new_data;
  int my_x, my_y;
  int num_neighbors;
  int neighbors_reported;
  int run_until;
};


Reduce :: Reduce(RedMsg *)
{
  count=0;
  next=0;
  numExpected=0;
  max=0.0;
}

void Reduce ::  Deposit(double val, Cell *id, int expected)
{
  //CPrintf("Deposit called in %d numExpected =%d \n", CMyPe(), expected);
  numExpected=expected + CNumSpanTreeChildren(CMyPe());
  ids[next++]=id;
  if (next >=50) {
	//CPrintf("ids OVERFLOW!!!!!!!!!!!!!!\n"); 
  }
  if (val > max) max=val;
  count++;
  //CPrintf("Deposit in %d count = %d num=%d\n", CMyPe(), count, numExpected);
  if (count == numExpected) {
        count=0;
	RedMsg *msg=(RedMsg *) new(MsgIndex(RedMsg)) RedMsg;
	msg->val=max;
	max=0.0;
	if (CMyPe()==0) {
		CBroadcastMsgBranch(Reduce, Recv_bcast, RedMsg, msg, thisgroup);
		return;
	}
	//CPrintf("%d Sending data to parent\n", CMyPe());
	CSendMsgBranch(Reduce, Recv_data, RedMsg, msg, thisgroup,CSpanTreeParent(CMyPe()));
  }
}

void Reduce :: Recv_data(RedMsg *m)
{
  //CPrintf("%d received data from child\n", CMyPe());
  if (m->val > max) max=m->val;
  count++;
  //CPrintf("recv_data in %d count = %d num=%d\n", CMyPe(), count, numExpected);
  if (count == numExpected) {
        count=0;
	RedMsg *msg=(RedMsg *) new(MsgIndex(RedMsg)) RedMsg;
	msg->val=max;
	max=0.0;

	if (CMyPe()==0) {
		CBroadcastMsgBranch(Reduce, Recv_bcast, RedMsg, msg, thisgroup);
		return;
	}
	//CPrintf("%d Sending data to parent\n", CMyPe());
	CSendMsgBranch(Reduce, Recv_data, RedMsg, msg, thisgroup,CSpanTreeParent(CMyPe()));
  }
}
	  
void Reduce ::  Recv_bcast(RedMsg *m)
{
  CPrintf("%d received bcast from root!!!!! Wall time= %f Timer= %f\n", CMyPe(), CmiWallTimer(), CmiTimer());
  for (int i=0;i<next;i++) {
	ids[i]->go_nogo();
  }
  max=0.0;
  next=0;
}

#include "jacobi.bot.h"
