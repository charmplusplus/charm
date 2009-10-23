#include <stdlib.h>
#include "blue.h"

int BCastID;
int ReduceID;
const int MAX_MESSAGES = 10;
const int NumBroadcasts = 10;	

extern "C" void BCastOcto(char*);
extern "C" void ReduceOcto(char*);

class OctoMsg;

void Octo(int, int, int, OctoMsg*, char*);
void SplitXYZ(int, int, int, OctoMsg*, char*);
bool HandleSpecial(int, int, int, OctoMsg*, char*);
// special cases
void Handle112(int, int, int, OctoMsg*, char*);
void Handle11X(int, int, int, OctoMsg*, char*);
void Handle122(int, int, int, OctoMsg*, char*);
void Handle12X(int, int, int, OctoMsg*, char*);
void Handle1XX(int, int, int, OctoMsg*, char*);
void Handle222(int, int, int, OctoMsg*, char*);
void Handle22X(int, int, int, OctoMsg*, char*);
void Handle2XX(int, int, int, OctoMsg*, char*);

class OctoMsg { 
  public:
    char core[CmiBlueGeneMsgHeaderSizeBytes];
    int x_min, y_min, z_min;
    int x_max, y_max, z_max;
    bool v;  // true if min_coord already visited
    int sender_x, sender_y, sender_z;

    OctoMsg() { }
    OctoMsg(int x1, int x2, int y1, int y2, int z1, int z2, bool visit) :
      x_min   (x1), 
      x_max   (x2),
      y_min   (y1),
      y_max   (y2),
      z_min   (z1),
      z_max   (z2),
      v       (visit)
    { 
    }

    /*
    friend CkOutStream& operator<<(CkOutStream& os, const OctoMsg& msg)
    {
      os <<"OctoMsg from " <<msg.sender_x <<" " <<msg.sender_y <<" " 
         <<msg.sender_z <<"->x(" <<msg.x_min <<"," <<msg.x_max <<")->y(" 
         <<msg.y_min <<"," <<msg.y_max <<")->z(" <<msg.z_min <<"," 
         <<msg.z_max <<")";
      return os;
    }
    */
    void *operator new(size_t s) { return CmiAlloc(s); }
    void operator delete(void* ptr) { CmiFree(ptr); }
};

class TimeMsg {
  public :
    char core[CmiBlueGeneMsgHeaderSizeBytes];
    double time;
    TimeMsg(double t) : time(t) { }
    void *operator new(size_t s) { return CmiAlloc(s); }
    void operator delete(void* ptr) { CmiFree(ptr); }
};

// for storing results of different tests
struct TimeRecord 
{
  int    test_num;
  double* max_time;
  double* start_time;

  TimeRecord() : test_num(0) {
    max_time = new double[NumBroadcasts];
    start_time = new double[NumBroadcasts];
    for (int i=0; i<NumBroadcasts; i++) {
      max_time[i] = start_time[i] = 0.0;
    }
  }

  ~TimeRecord() {
    delete [] max_time;
    delete [] start_time;
  }
     
  void Print(char* info) {
    //print result
    double average = 0.0;
    int sizeX, sizeY, sizeZ;
    BgGetSize(&sizeX, &sizeY, &sizeZ);
    int numComm = BgGetNumCommThread();
    int numWork = BgGetNumWorkThread();

    CmiPrintf("\nResults for %d by %d by %d with %d comm %d work\n\n",
              sizeX, sizeY, sizeZ, numComm, numWork);
    CmiPrintf("-------------------------------------------------------------\n"
              "Iter No:    StartTime           EndTime          TotalTime   \n"
              "-------------------------------------------------------------\n");
    for (int i=0; i<NumBroadcasts; i++) {
      CmiPrintf("    %d         %f               %f           %f\n",
                i, start_time[i], max_time[i], max_time[i] - start_time[i]);
      average += max_time[i] - start_time[i];
    }
    CmiPrintf("-------------------------------------------------------------\n"
              "Average BroadCast Time:  	            %f\n"
              "-------------------------------------------------------------\n",
              average/NumBroadcasts);
    BgShutdown();
  }
};

struct OctoData 
{
  OctoMsg     messages[MAX_MESSAGES];
  int         dest_x[MAX_MESSAGES];
  int         dest_y[MAX_MESSAGES];
  int         dest_z[MAX_MESSAGES];
  int         root_x, root_y, root_z;
  int         parent_x, parent_y, parent_z;
  TimeRecord* record;
  
  OctoData(int x, int y, int z): 
    root_x       (x),
    root_y       (y),
    root_z       (z),
    record       (NULL)
 { }
};

BnvStaticDeclare(int, num_messages)
BnvStaticDeclare(double, max_time)
BnvStaticDeclare(int, reduce_count)

void BgEmulatorInit(int argc, char** argv)
{
  if (argc < 6) { 
    CmiPrintf("Usage: octo <x> <y> <z> <numComm> <numWork> \n"); 
    BgShutdown();
  }
    
  BgSetSize(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
  BgSetNumCommThread(atoi(argv[4]));
  BgSetNumWorkThread(atoi(argv[5]));

}

void BgWorkThreadInit(int argc, char** argv)
{
}

void BgNodeStart(int argc, char** argv)
{
  BCastID = BgRegisterHandler(BCastOcto);
  ReduceID  = BgRegisterHandler(ReduceOcto);

  int x, y, z;
  BgGetMyXYZ(&x, &y, &z);
  int numBgX, numBgY, numBgZ;
  BgGetSize(&numBgX, &numBgY, &numBgZ);

  BnvInitialize(int, num_messages);
  BnvAccess(num_messages) = -1;
  BnvInitialize(double, max_time);
  BnvAccess(max_time) = 0.0;
  BnvInitialize(int, reduce_count);
  BnvAccess(reduce_count) = 0;
 
  
  int center_x = (numBgX == 2) ? 0 : numBgX/2;
  int center_y = (numBgY == 2) ? 0 : numBgY/2;
  int center_z = (numBgZ == 2) ? 0 : numBgZ/2;
  OctoData* data = new OctoData(center_x, center_y, center_z);
  BgSetNodeData((char*)data);

  // check to see if center node
  if (x == center_x && y == center_y && z == center_z) 
  {
    data->record = new TimeRecord();
    OctoMsg* msg = new OctoMsg(0, numBgX-1, 0, numBgY-1, 0, numBgZ-1, false);
    BgSendLocalPacket(ANYTHREAD, BCastID, LARGE_WORK, sizeof(OctoMsg), 
                      (char*)msg);
  }
}

void BCastOcto(char* info) {
  int x, y, z;
  BgGetMyXYZ(&x, &y, &z);

  OctoMsg* m = (OctoMsg*)info;
  OctoData* d = (OctoData*)BgGetNodeData();
  
  // CmiPrintf("BCastOcto at %d %d %d range x=(%d,%d) y=(%d,%d) z=(%d,%d)\n", 
  //           x, y, z, m->x_min, m->x_max, m->y_min, m->y_max, m->z_min, 
  //           m->z_max);

  if (x == d->root_x && y == d->root_y && z == d->root_z) {
    if (d->record == NULL) {
      CmiPrintf("Error, root node should have timing info\n");
      BgShutdown();
      return;
    }
    TimeRecord* r = d->record;
    r->start_time[r->test_num] = BgGetTime();
    // CmiPrintf("Starting new broadcast at %f\n", r->start_time[r->test_num]);
  }

  // if first time through, calculate correct nodes to send
  if (BnvAccess(num_messages) == -1) { 
    d->parent_x = m->sender_x;
    d->parent_y = m->sender_y;
    d->parent_z = m->sender_z;
  
    SplitXYZ(x, y, z, m, info); 
  }

  // once calculated, send to nodes
  if (BnvAccess(num_messages) <= 0) {
    // this is a terminal node, so start reduction
    TimeMsg* parent = new TimeMsg(BgGetTime());
    // CmiPrintf("  Terminal node, time %f to %d %d %d\n", 
    //           parent->time, d->parent_x, d->parent_y, d->parent_z);
    BgSendPacket(d->parent_x, d->parent_y, d->parent_z, 
                 ANYTHREAD, ReduceID, SMALL_WORK, sizeof(TimeMsg), 
                 (char*)parent);
  }
  else {
    // CmiPrintf("  There are %d messages\n", d->num_messages);
    for (int i=0; i<BnvAccess(num_messages); i++) {
      OctoMsg* msg = new OctoMsg(d->messages[i]);
      // CmiPrintf("    Sending message %d/%d to %d %d %d\n", i+1, 
      //           d->num_messages, d->dest_x[i], d->dest_y[i], d->dest_z[i]);
      BgSendPacket(d->dest_x[i], d->dest_y[i], d->dest_z[i], 
                   ANYTHREAD, BCastID, SMALL_WORK, sizeof(OctoMsg), (char*)msg);
    }
  }
}

void ReduceOcto(char* info) 
{
  int x, y, z;
  BgGetMyXYZ(&x, &y, &z);

  OctoData* d = (OctoData*)BgGetNodeData();
  TimeMsg* msg = (TimeMsg*)info;

  BnvAccess(reduce_count)++;

  //CmiPrintf("ReduceOcto at node %d %d %d message %d/%d with time %f "
  //          "max time is %f\n", x, y, z, BnvAccess(reduce_count), 
  //          BnvAccess(num_messages), msg->time, BnvAccess(max_time));

  if (msg->time > BnvAccess(max_time)) { BnvAccess(max_time) = msg->time; }

  // check to see if all children have been heard from
  if (BnvAccess(reduce_count) >= BnvAccess(num_messages)) {
    BnvAccess(reduce_count) = 0;

    if (x == d->root_x && y == d->root_y && z == d->root_z) {
      // this is the root node
      TimeRecord* r = d->record;
      r->max_time[r->test_num] = BnvAccess(max_time);
      r->test_num++;
      if (r->test_num < NumBroadcasts) {
        // start bcast all over again
        int numBgX, numBgY, numBgZ;
        BgGetSize(&numBgX, &numBgY, &numBgZ);
        OctoMsg* start = 
          new OctoMsg(0, numBgX-1, 0, numBgY-1, 0, numBgZ-1, false);
        BgSendLocalPacket(ANYTHREAD, BCastID, SMALL_WORK, sizeof(OctoMsg), 
                          (char*)start);
      }
      else {
        // print results and quit
        r->Print(info);
        BgShutdown();
        return;
      }
    }
    else {
      // this is not the root node
      TimeMsg* parent = new TimeMsg(BnvAccess(max_time));
      BgSendPacket(d->parent_x, d->parent_y, d->parent_z, ANYTHREAD, ReduceID,
                   SMALL_WORK, sizeof(TimeMsg), (char*)parent);
      BnvAccess(max_time) = 0;
    }
  }
}

void SplitXYZ
(
  int         x,     // x pos of node
  int         y,     // y pos of node
  int         z,     // z pos of node
  OctoMsg*    m,     // message with limit data
  char* info
)
{
  // check terminal case
  if (x == m->x_min && x == m->x_max &&
      y == m->y_min && y == m->y_max &&
      z == m->z_min && z == m->z_max)
  {
    return;
  }
  
  if (HandleSpecial(x, y, z, m, info)) { return; }

  int del_x = m->x_max - m->x_min;
  int del_y = m->y_max - m->y_min;
  int del_z = m->z_max - m->z_min;

  // calculate new midpoints
  int x_lo = (m->x_min + x - 1)/2;  int x_hi = (m->x_max + x)/2;
  int y_lo = (m->y_min + y - 1)/2;  int y_hi = (m->y_max + y)/2;
  int z_lo = (m->z_min + z - 1)/2;  int z_hi = (m->z_max + z)/2;
  
  OctoMsg* new_msg = NULL;

  // x_lo, y_lo, z_lo
  new_msg = new OctoMsg(m->x_min, x-1, m->y_min, y-1, m->z_min, z-1, m->v);
  if (x_lo == m->x_min && y_lo == m->y_min && z_lo == m->z_min && m->v) {
    SplitXYZ(x_lo, y_lo, z_lo, new_msg, info);
    delete new_msg;
  }
  else { Octo(x_lo, y_lo, z_lo, new_msg, info); }
  // x_lo, y_lo, z_hi
  new_msg = new OctoMsg(m->x_min, x-1, m->y_min, y-1, z, m->z_max, false);
  Octo(x_lo, y_lo, z_hi, new_msg, info);
  // x_lo, y_hi, z_lo
  new_msg = new OctoMsg(m->x_min, x-1, y, m->y_max, m->z_min, z-1, false);
  Octo(x_lo, y_hi, z_lo, new_msg, info);
  // x_lo, y_hi, z_hi
  new_msg = new OctoMsg(m->x_min, x-1, y, m->y_max, z, m->z_max, false);
  Octo(x_lo, y_hi, z_hi, new_msg, info);
  // x_hi, y_lo, z_lo
  new_msg = new OctoMsg(x, m->x_max, m->y_min, y-1, m->z_min, z-1, false);
  Octo(x_hi, y_lo, z_lo, new_msg, info);
  // x_hi, y_lo, z_hi
  new_msg = new OctoMsg(x, m->x_max, m->y_min, y-1, z, m->z_max, false);
  Octo(x_hi, y_lo, z_hi, new_msg, info);
  // x_hi, y_hi, z_lo
  new_msg = new OctoMsg(x, m->x_max, y, m->y_max, m->z_min, z-1, false);
  Octo(x_hi, y_hi, z_lo, new_msg, info);
  // x_hi, y_hi, z_hi (x, y, z) 
  new_msg = new OctoMsg(x, m->x_max, y, m->y_max, z, m->z_max, true);
  if (x_hi == x && y_hi == y && z_hi == z) {
    // just call local function instead of passing message
    SplitXYZ(x_hi, y_hi, z_hi, new_msg, info);
    delete new_msg;
  }
  else { Octo(x_hi, y_hi, z_hi, new_msg, info); }
}

void Octo
(
  int         x,     // x pos of node
  int         y,     // y pos of node
  int         z,     // z pos of node
  OctoMsg*    m,     // message with limit data
  char* info
)
{
  OctoData* d = (OctoData*)BgGetNodeData();
  BgGetMyXYZ(&m->sender_x, &m->sender_y, &m->sender_z);

  // save the message if desired
  if (BnvAccess(num_messages) == -1) { BnvAccess(num_messages)++; }
  if (BnvAccess(num_messages) < MAX_MESSAGES) {
    int i = BnvAccess(num_messages);
    d->dest_x[i] = x;
    d->dest_y[i] = y;
    d->dest_z[i] = z;
    d->messages[i] = *m;
    BnvAccess(num_messages)++;
  }
  else { 
    CmiPrintf("ERR: No room to write message on node %d %d %d\n",
              m->sender_x, m->sender_y, m->sender_z);
    BgShutdown();
    return;
  }
}

bool HandleSpecial
(
  int         x,     // x pos of node
  int         y,     // y pos of node
  int         z,     // z pos of node
  OctoMsg*    m,     // message with limit data
  char* info
)
{
  int del_x = m->x_max - m->x_min;
  int del_y = m->y_max - m->y_min;
  int del_z = m->z_max - m->z_min;

  if (del_x >= 2 && del_y >= 2 && del_z >= 2) {
    return false;
  }

  if ((del_x == 1 && del_y == 0 && del_z == 0) ||
      (del_x == 0 && del_y == 1 && del_z == 0) ||
      (del_x == 0 && del_y == 0 && del_z == 1))
  {
    Handle112(x, y, z, m, info);
  }
  else if ((del_x >  1 && del_y == 0 && del_z == 0) ||
           (del_x == 0 && del_y  > 1 && del_z == 0) ||
           (del_x == 0 && del_y == 0 && del_z  > 1))
  {
    Handle11X(x, y, z, m, info);
  }
  else if ((del_x == 1 && del_y == 1 && del_z == 0) ||
           (del_x == 1 && del_y == 0 && del_z == 1) ||
           (del_x == 0 && del_y == 1 && del_z == 1))
  {
    Handle122(x, y, z, m, info);
  }
  else if ((del_x == 0 && del_y == 1 && del_z  > 1) ||
           (del_x == 0 && del_y  > 1 && del_z == 1) ||
           (del_x == 1 && del_y == 0 && del_z  > 1) ||
           (del_x == 1 && del_y  > 1 && del_z == 0) ||
           (del_x  > 1 && del_y == 0 && del_z == 1) ||
           (del_x  > 1 && del_y == 1 && del_z == 0)) 
  {
    Handle12X(x, y, z, m, info);
  }
  else if ((del_x == 0 && del_y  > 1 && del_z  > 1) ||
           (del_x  > 1 && del_y == 0 && del_z  > 1) ||
           (del_x  > 1 && del_y  > 1 && del_z == 0))
  {
    Handle1XX(x, y, z, m, info);
  }
  else if (del_x == 1 && del_y == 1 && del_z == 1) {
    Handle222(x, y, z, m, info);
  }
  else if ((del_x == 1 && del_y == 1 && del_z  > 1) ||
           (del_x == 1 && del_y  > 1 && del_z == 1) ||
           (del_x  > 1 && del_y == 1 && del_z == 1)) 
  {
    Handle22X(x, y, z, m, info);
  }
  else if ((del_x == 1 && del_y  > 1 && del_z  > 1) ||
           (del_x  > 1 && del_y == 1 && del_z  > 1) ||
           (del_x  > 1 && del_y  > 1 && del_z == 1)) 
  {
    Handle2XX(x, y, z, m, info);
  }
  else {
    CmiPrintf("ERR: COULDN'T HANDLE %d x %d x %d\n", del_x+1, del_y+1, del_z+1);
    BgShutdown();
    return false;
  }
  
  return true;
}

void Handle112(int x, int y, int z, OctoMsg* m, char* info)
{
  // 2x1x1 -> only 2nd x hasn't been touched
  // 1x2x1 -> only 2nd y hasn't been touched
  // 1x1x2 -> only 2nd z hasn't been touched

  int del_x = m->x_max - m->x_min;
  int del_y = m->y_max - m->y_min;
  int del_z = m->z_max - m->z_min;

  OctoMsg* new_msg = 
    new OctoMsg(x+del_x, x+del_x, y+del_y, y+del_y, z+del_z, z+del_z, false);
  Octo(x+del_x, y+del_y, z+del_z, new_msg, info);
}

void Handle11X(int x, int y, int z, OctoMsg* m, char* info)
{
  int x_mod = ((m->x_max - m->x_min) > 1) ? 1 : 0;
  int y_mod = ((m->y_max - m->y_min) > 1) ? 1 : 0;
  int z_mod = ((m->z_max - m->z_min) > 1) ? 1 : 0; 
    
  // calculate new midpoints
  int x_lo = (m->x_min + x - x_mod)/2;  int x_hi = (m->x_max + x)/2;
  int y_lo = (m->y_min + y - y_mod)/2;  int y_hi = (m->y_max + y)/2;
  int z_lo = (m->z_min + z - z_mod)/2;  int z_hi = (m->z_max + z)/2;

  // hit high node
  OctoMsg* new_msg = new OctoMsg(x, m->x_max, y, m->y_max, z, m->z_max, true);
  if (x == x_hi && y == y_hi && z == z_hi) {
    SplitXYZ(x_hi, y_hi, z_hi, new_msg, info);
    delete new_msg;
  }
  else { Octo(x_hi, y_hi, z_hi, new_msg, info); }

  // hit low node if it hasn't been hit yet
  int new_x_width = (x - x_mod - m->x_min + 1);
  int new_y_width = (y - y_mod - m->y_min + 1);
  int new_z_width = (z - z_mod - m->z_min + 1);
  
  new_msg = new OctoMsg(m->x_min, x-x_mod, m->y_min, y-y_mod, 
                        m->z_min, z-z_mod, m->v);

  // error handle all different cases
  if (new_x_width * new_y_width * new_z_width == 1) {
    if (!m->v) { Octo(x_lo, y_lo, z_lo, new_msg, info); }
    else { delete new_msg; }
  }
  else if (x == x_lo && y == y_lo && z == z_lo) {
    SplitXYZ(x_lo, y_lo, z_lo, new_msg, info);
    delete new_msg;
  }
  else if (new_x_width * new_y_width * new_z_width != 0) { 
    if (x_lo == m->x_min && y_lo == m->y_min && z_lo == m->z_min && m->v) {
      SplitXYZ(x_lo, y_lo, z_lo, new_msg, info);
      delete new_msg;
    }
    else { Octo(x_lo, y_lo, z_lo, new_msg, info); }
  }
}

void Handle122(int x, int y, int z, OctoMsg* m, char* info)
{
  int del_x = m->x_max - m->x_min;
  int del_y = m->y_max - m->y_min;
  int del_z = m->z_max - m->z_min;

  OctoMsg* new_msg = NULL;
  
  // 2x2x1 -> split along x axiz and handle it
  // 2x1x2 -> split along x axis and handle it
  if (del_z == 0 || del_y == 0)
  {
    new_msg = new OctoMsg(x, x, m->y_min, m->y_max, m->z_min, m->z_max, true);
    SplitXYZ(x, y, z, new_msg, info); 
    delete new_msg;

    new_msg = 
      new OctoMsg(x+1, x+1, m->y_min, m->y_max, m->z_min, m->z_max, false);
    Octo(x+1, y, z, new_msg, info);
  }
  // 1x2x2 -> split along y axis and handle it
  else if (del_x == 0) {
    new_msg = new OctoMsg(x, x, y, y, m->z_min, m->z_max, true);
    SplitXYZ(x, y, z, new_msg, info);
    delete new_msg;

    new_msg = new OctoMsg(x, x, y+1, y+1, m->z_min, m->z_max, false);
    Octo(x, y+1, z, new_msg, info);
  }
}

void Handle12X(int x, int y, int z, OctoMsg* m, char* info)
{
  int x_mod = ((m->x_max - m->x_min) > 1) ? 1 : 0;
  int y_mod = ((m->y_max - m->y_min) > 1) ? 1 : 0;
  int z_mod = ((m->z_max - m->z_min) > 1) ? 1 : 0; 
    
  // calculate new midpoints
  int x_lo = (m->x_min + x - x_mod)/2;  int x_hi = (m->x_max + x)/2;
  int y_lo = (m->y_min + y - y_mod)/2;  int y_hi = (m->y_max + y)/2;
  int z_lo = (m->z_min + z - z_mod)/2;  int z_hi = (m->z_max + z)/2;

  // first split high side
  OctoMsg* new_msg = new OctoMsg(x, m->x_max, y, m->y_max, z, m->z_max, true);
  if (x == x_hi && y == y_hi && z == z_hi) {
    SplitXYZ(x_hi, y_hi, z_hi, new_msg, info);
    delete new_msg;
  }
  else { Octo(x_hi, y_hi, z_hi, new_msg, info); }

  int del_x = m->x_max - m->x_min;
  int del_y = m->y_max - m->y_min;
  int del_z = m->z_max - m->z_min;

  // hit low node if it hasn't been hit yet
  int new_x_width = del_x > 1 ? (x - x_mod - m->x_min + 1) : del_x + 1;
  int new_y_width = del_y > 1 ? (y - y_mod - m->y_min + 1) : del_y + 1;
  int new_z_width = del_z > 1 ? (z - z_mod - m->z_min + 1) : del_z + 1;
  
  // adjust mods
  x_mod = ((m->x_max - m->x_min) == 1) ? -1 : x_mod;
  y_mod = ((m->y_max - m->y_min) == 1) ? -1 : y_mod;
  z_mod = ((m->z_max - m->z_min) == 1) ? -1 : z_mod; 
  new_msg = new OctoMsg(m->x_min, x-x_mod, m->y_min, y-y_mod, 
                        m->z_min, z-z_mod, m->v);

  // error handle all different cases
  if (new_x_width * new_y_width * new_z_width == 2) {
    if (!m->v) { Octo(x_lo, y_lo, z_lo, new_msg, info); }
    else { 
      SplitXYZ(x_lo, y_lo, z_lo, new_msg, info);
      delete new_msg; 
    }
  }
  else if (x == x_lo && y == y_lo && z == z_lo) {
    SplitXYZ(x_lo, y_lo, z_lo, new_msg, info);
    delete new_msg;
  }
  else if (new_x_width * new_y_width * new_z_width != 0) { 
    if (x_lo == m->x_min && y_lo == m->y_min && z_lo == m->z_min && m->v) {
      SplitXYZ(x_lo, y_lo, z_lo, new_msg, info);
      delete new_msg;
    }
    else { Octo(x_lo, y_lo, z_lo, new_msg, info); }
  }
}

void Handle1XX(int x, int y, int z, OctoMsg* m, char* info)
{
  int del_x = m->x_max - m->x_min;
  int del_y = m->y_max - m->y_min;
  int del_z = m->z_max - m->z_min;

  int x_mod = ((m->x_max - m->x_min) > 1) ? 1 : 0;
  int y_mod = ((m->y_max - m->y_min) > 1) ? 1 : 0;
  int z_mod = ((m->z_max - m->z_min) > 1) ? 1 : 0; 

  // calculate new midpoints
  int x_lo = (m->x_min + x - x_mod)/2;  int x_hi = (m->x_max + x)/2;
  int y_lo = (m->y_min + y - y_mod)/2;  int y_hi = (m->y_max + y)/2;
  int z_lo = (m->z_min + z - z_mod)/2;  int z_hi = (m->z_max + z)/2;

  OctoMsg* new_msg = 
    new OctoMsg(x, m->x_max, y, m->y_max, z, m->z_max, true);
  if (x == x_hi && y == y_hi && z == z_hi) {
    SplitXYZ(x_hi, y_hi, z_hi, new_msg, info);
    delete new_msg;
  }
  else { Octo(x_hi, y_hi, z_hi, new_msg, info); }
    
  // hit low node if it hasn't been hit yet
  int new_x_width = (x-x_mod - m->x_min + 1);
  int new_y_width = (y-y_mod - m->y_min + 1);
  int new_z_width = (z-z_mod - m->z_min + 1);
  
  new_msg = new OctoMsg(m->x_min, x-x_mod, m->y_min, y-y_mod, 
                        m->z_min, z-z_mod, m->v);

  // error handle all different cases
  if (new_x_width * new_y_width * new_z_width == 1) {
    if (!m->v) { Octo(x_lo, y_lo, z_lo, new_msg, info); }
    else { delete new_msg; }
  }
  else if (x == x_lo && y == y_lo && z == z_lo) {
    SplitXYZ(x_lo, y_lo, z_lo, new_msg, info);
    delete new_msg;
  }
  else { 
    if (x_lo == m->x_min && y_lo == m->y_min && z_lo == m->z_min && m->v) {
      SplitXYZ(x_lo, y_lo, z_lo, new_msg, info);
      delete new_msg;
    }
    else { Octo(x_lo, y_lo, z_lo, new_msg, info); }
  }

  // calculate new mods
  x_mod = (x - m->x_min - 1);  x_mod = x_mod > 0 ? x_mod : 0;
  y_mod = (y - m->y_min - 1);  y_mod = y_mod > 0 ? y_mod : 0;
  z_mod = (z - m->z_min - 1);  z_mod = z_mod > 0 ? z_mod : 0;

  // 1xXxX
  if (del_x == 0) {
    new_msg = 
      new OctoMsg(x, x, y, m->y_max, m->z_min, m->z_min + z_mod, false);
    Octo(x, y_hi, z_lo, new_msg, info);
    
    new_msg = 
      new OctoMsg(x, x, m->y_min, m->y_min + y_mod, z, m->z_max, false);
    Octo(x, y_lo, z_hi, new_msg, info);
  }
  // Xx1xX
  else if (del_y == 0) {
    new_msg = 
      new OctoMsg(x, m->x_max, y, y, m->z_min, m->z_min + z_mod, false);
    Octo(x_hi, y, z_lo, new_msg, info);
    
    new_msg = 
      new OctoMsg(m->x_min, m->x_min + x_mod, y, y, z, m->z_max, false);
    Octo(x_lo, y, z_hi, new_msg, info);
  }
  // XxXx1
  else { // (del_z == 0)
    new_msg = 
      new OctoMsg(x, m->x_max, m->y_min, m->y_min + y_mod, z, z, false);
    Octo(x_hi, y_lo, z, new_msg, info);
    
    new_msg = 
      new OctoMsg(m->x_min, m->x_min + x_mod, y, m->y_max, z, z, false);
    Octo(x_lo, y_hi, z, new_msg, info);
  }
}

void Handle222(int x, int y, int z, OctoMsg* m, char* info)
{
  int del_x = m->x_max - m->x_min;
  int del_y = m->y_max - m->y_min;
  int del_z = m->z_max - m->z_min;

  // 2x2x2 -> split along z axis and handle it
  OctoMsg* new_msg = 
    new OctoMsg(m->x_min, m->x_max, m->y_min, m->y_max, z, z, true);
  SplitXYZ(x, y, z, new_msg, info);
  delete new_msg;

  new_msg = 
    new OctoMsg(m->x_min, m->x_max, m->y_min, m->y_max, z+1, z+1, m->v);
  Octo(x, y, z+1, new_msg, info);
}

void Handle22X(int x, int y, int z, OctoMsg* m, char* info)
{
  int del_x = m->x_max - m->x_min;
  int del_y = m->y_max - m->y_min;
  int del_z = m->z_max - m->z_min;

  // calculate new midpoints
  int x_lo = (m->x_min + x - 1)/2;  int x_hi = (m->x_max + x)/2;
  int y_lo = (m->y_min + y - 1)/2;  int y_hi = (m->y_max + y)/2;
  int z_lo = (m->z_min + z - 1)/2;  int z_hi = (m->z_max + z)/2;

  OctoMsg* new_msg = NULL;

  // 2x2xX
  if (del_z != 1) {
    new_msg = 
      new OctoMsg(m->x_min, m->x_max, m->y_min, m->y_max, z, m->z_max, true);
    if (z == z_hi) {
      SplitXYZ(x, y, z_hi, new_msg, info);
      delete new_msg;
    }
    else { Octo(x, y, z_hi, new_msg, info); }

    new_msg = 
      new OctoMsg(m->x_min, m->x_max, m->y_min, m->y_max, m->z_min, z-1, m->v);
    if (z_lo == m->z_min && m->v) {
      SplitXYZ(x, y, z_lo, new_msg, info);
      delete new_msg;
    }
    else { Octo(x, y, z_lo, new_msg, info); }
  }
  // 2xXx2
  else if (del_y != 1) {
    new_msg = 
      new OctoMsg(m->x_min, m->x_max, y, m->y_max, m->z_min, m->z_max, true);
    if (y == y_hi) {
      SplitXYZ(x, y_hi, z, new_msg, info); 
      delete new_msg;
    }
    else { Octo(x, y_hi, z, new_msg, info); }

    new_msg = 
      new OctoMsg(m->x_min, m->x_max, m->y_min, y-1, m->z_min, m->z_max, m->v);
    if (y_lo == m->y_min && m->v) {
      SplitXYZ(x, y_lo, z, new_msg, info);
      delete new_msg;
    }
    else { Octo(x, y_lo, z, new_msg, info); }
  }
  // Xx2x2
  else { // (del_x != 1)
    new_msg = 
      new OctoMsg(x, m->x_max, m->y_min, m->y_max, m->z_min, m->z_max, true);
    if (x == x_hi) {
      SplitXYZ(x_hi, y, z, new_msg, info); 
      delete new_msg;
    }
    else { Octo(x_hi, y, z, new_msg, info); }

    new_msg = 
      new OctoMsg(m->x_min, x-1, m->y_min, m->y_max, m->z_min, m->z_max, m->v);
    if (x_lo == m->x_min && m->v) {
      SplitXYZ(x_lo, y, z, new_msg, info);
      delete new_msg;
    }
    else { Octo(x_lo, y, z, new_msg, info); }
  }
}

void Handle2XX(int x, int y, int z, OctoMsg* m, char* info)
{
  int del_x = m->x_max - m->x_min;
  int del_y = m->y_max - m->y_min;
  int del_z = m->z_max - m->z_min;

  // calculate new midpoints
  int x_lo = (m->x_min + x - 1)/2;  int x_hi = (m->x_max + x)/2;
  int y_lo = (m->y_min + y - 1)/2;  int y_hi = (m->y_max + y)/2;
  int z_lo = (m->z_min + z - 1)/2;  int z_hi = (m->z_max + z)/2;

  OctoMsg* new_msg = NULL;

  // get upper corner
  new_msg = new OctoMsg(x, m->x_max, y, m->y_max, z, m->z_max, true);
  if (x == x_hi && y == y_hi && z == z_hi) {
    SplitXYZ(x_hi, y_hi, z_hi, new_msg, info);
    delete new_msg;
  }
  else { Octo(x_hi, y_hi, z_hi, new_msg, info); }

  // 2xXxX 
  if (del_x == 1) {
    // lower corner
    new_msg = new OctoMsg(x, x+1, m->y_min, y-1, m->z_min, z-1, m->v);
    if (x == m->x_min && y_lo == m->y_min && z_lo == m->z_min && m->v) {
      SplitXYZ(x, y_lo, z_lo, new_msg, info); 
      delete new_msg;
    }
    else { Octo(x, y_lo, z_lo, new_msg, info); }

    // y_hi, z_lo
    new_msg = new OctoMsg(x, x+1, y, m->y_max, m->z_min, z-1, false);
    Octo(x, y_hi, z_lo, new_msg, info);

    // y_lo, z_hi
    new_msg = new OctoMsg(x, x+1, m->y_min, y-1, z, m->z_max, false);
    Octo(x, y_lo, z_hi, new_msg, info);
  }
  // Xx2xX
  else if (del_y == 1) {
    // lower corner
    new_msg = new OctoMsg(m->x_min, x-1, y, y+1, m->z_min, z-1, m->v);
    if (x_lo == m->x_min && y == m->y_min && z_lo == m->z_min && m->v) {
      SplitXYZ(x_lo, y, z_lo, new_msg, info);
      delete new_msg;
    }
    else { Octo(x_lo, y, z_lo, new_msg, info); }

    // x_hi, z_lo
    new_msg = new OctoMsg(x, m->x_max, y, y+1, m->z_min, z-1, false);
    Octo(x_hi, y, z_lo, new_msg, info);

    // x_lo, z_hi
    new_msg = new OctoMsg(m->x_min, x-1, y, y+1, z, m->z_max, false);
    Octo(x_lo, y, z_hi, new_msg, info);
  }
  // XxXx2
  else if (del_z == 1) {
    // lower corner
    new_msg = new OctoMsg(m->x_min, x-1, m->y_min, y-1, z, z+1, m->v);
    if (x_lo == m->x_min && y_lo == m->y_min && z == m->z_min && m->v) {
      SplitXYZ(x_lo, y_lo, z, new_msg, info);
      delete new_msg;
    }
    else { Octo(x_lo, y_lo, z, new_msg, info); }

    // x_hi, y_lo
    new_msg = new OctoMsg(x, m->x_max, m->y_min, y-1, z, z+1, false);
    Octo(x_hi, y_lo, z, new_msg, info); 

    // x_lo, y_hi
    new_msg = new OctoMsg(m->x_min, x-1, y, m->y_max, z, z+1, false);
    Octo(x_lo, y_hi, z, new_msg, info);
  }
}

