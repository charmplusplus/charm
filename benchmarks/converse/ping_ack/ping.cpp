#include <stdio.h>
#include <math.h>
#include <time.h>
#include <converse.h>

CpvDeclare(int, bigmsg_index);
CpvDeclare(int, ackmsg_index);
CpvDeclare(int, shortmsg_index);
CpvDeclare(int, stop_index);
CpvDeclare(int, msg_size);
CpvDeclare(int, trial);               // increments per trial, gets set to 0 at the start of a new msg size
CpvDeclare(int, round);               // increments per msg size
CpvDeclare(int, warmup_flag);         // 1 when in warmup round, 0 when not
CpvDeclare(int, recv_count);
CpvDeclare(int, ack_count);
CpvDeclare(double, total_time);
CpvDeclare(double, process_time);
CpvDeclare(double, send_time);

#define MSG_COUNT 100
#define nMSG_SIZE 3                   // if the msg_sizes are hard_coded, this should be the same as the length of the hard coded array
#define nTRIALS_PER_SIZE 10
#define CALCULATION_PRECISION 0.0001  // the decimal place that the output data is rounded to

double total_time[nTRIALS_PER_SIZE];  // times are stored in us
double process_time[nTRIALS_PER_SIZE];
double send_time[nTRIALS_PER_SIZE];


int msg_sizes[nMSG_SIZE] = {56, 4096, 65536}; // hard coded msg_size values



typedef struct myMsg
{
  char header[CmiMsgHeaderSizeBytes];
  int payload[1];
} *message;

// helper functions

double round_to(double val, double precision) {
  return round(val / precision) * precision;
}

double get_average(double arr[]) {
  double tot = 0;
  for (int i = 0; i < nTRIALS_PER_SIZE; ++i) tot += arr[i];
  return (round_to(tot, CALCULATION_PRECISION) / nTRIALS_PER_SIZE);

}

double get_stdev(double arr[]) {
  double stdev = 0.0;
  double avg = get_average(arr);
  for (int i = 0; i < nTRIALS_PER_SIZE; ++i)
    stdev += pow(arr[i] - avg, 2);
  stdev = sqrt(stdev / nTRIALS_PER_SIZE);
  return stdev;
}

double get_max(double arr[]) {
  double max = arr[0];
  for (int i = 1; i < nTRIALS_PER_SIZE; ++i)
                if (arr[i] > arr[0]) max = arr[i];
        return max;
}


void print_results() {
  if (!CpvAccess(warmup_flag)) {
    CmiPrintf("msg_size\n%d\n", CpvAccess(msg_size));
    for (int i = 0; i < nTRIALS_PER_SIZE; ++i) {
      // DEBUG: print without trial number:
      // CmiPrintf("%f\n%f\n%f\n", send_time[i], process_time[i], total_time[i]);

      // DEBUG: print with trial number:
      // CmiPrintf("%d %f\n  %f\n  %f\n", i, send_time[i], process_time[i], total_time[i]);
    }
    // print data:
    CmiPrintf("Format: {#PEs},{msg_size},{averages*3},{stdevs*3},{maxs*3}\n");
    CmiPrintf("DATA,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n", CmiNumPes(), CpvAccess(msg_size), get_average(send_time), get_average(process_time), get_average(total_time),
                                get_stdev(send_time), get_stdev(process_time), get_stdev(total_time), get_max(send_time), get_max(process_time), get_max(total_time));
                
                
  } else {
    if (CpvAccess(round) == nMSG_SIZE - 1)  // if this is the end of the warmup round
      CmiPrintf("Warm up Done!\n");

    // DEBUG: Print what msg_size the warmup round is on
    // else                               // otherwise move to the next msg size
    //  CmiPrintf("Warming up msg_size %d\n", CpvAccess(msg_size));
  }
}

void stop(void *msg)
{
  CsdExitScheduler();
}

void send_msg() {
  double start_time, crt_time;
  struct myMsg *msg;
  //  CmiPrintf("\nSending msg fron pe%d to pe%d\n",CmiMyPe(), CmiNumPes()/2+CmiMyPe());
  CpvAccess(process_time) = 0.0;
  CpvAccess(send_time) = 0.0;
  CpvAccess(total_time) = CmiWallTimer();
  for(int k = 0; k < MSG_COUNT; k++) {
    crt_time = CmiWallTimer();
    msg = (message)CmiAlloc(CpvAccess(msg_size));

    // Fills payload with ints
    for (int i = 0; i < (CpvAccess(msg_size) - CmiMsgHeaderSizeBytes) / sizeof(int); ++i) msg->payload[i] = i;
    
    // DEBUG: Print ints stored in payload
    // for (int i = 0; i < (CpvAccess(msg_size) - CmiMsgHeaderSizeBytes) / sizeof(int); ++i) CmiPrintf("%d ", msg->payload[i]);
    // CmiPrintf("\n");

    CmiSetHandler(msg, CpvAccess(bigmsg_index));
    CpvAccess(process_time) = CmiWallTimer() - crt_time + CpvAccess(process_time);
    start_time = CmiWallTimer();
    //Send from my pe-i on node-0 to q+i on node-1
    CmiSyncSendAndFree(CmiNumPes() / 2 + CmiMyPe(), CpvAccess(msg_size), msg);
    CpvAccess(send_time) = CmiWallTimer() - start_time + CpvAccess(send_time);
  }
}



void shortmsg_handler(void *vmsg) {
  message smsg = (message)vmsg;
  CmiFree(smsg);
  if (!CpvAccess(warmup_flag)) {     // normal round handling
    if (CpvAccess(trial) == nTRIALS_PER_SIZE) { // if we have run the current msg size for nTRIALS
      CpvAccess(round) = CpvAccess(round) + 1;
      CpvAccess(trial) = 0;
      CpvAccess(msg_size) = msg_sizes[CpvAccess(round)];
    } 
  } else {   // warmup round handling
    if (CpvAccess(round) == nMSG_SIZE - 1) {  // if this is the end of the warmup round
      CpvAccess(round) = 0;
      CpvAccess(msg_size) = msg_sizes[0];
      CpvAccess(warmup_flag) = 0;
    } else {                                  // otherwise warm up the next msg size
      CpvAccess(round) = CpvAccess(round) + 1;
      CpvAccess(msg_size) = msg_sizes[CpvAccess(round)];
    }
    CpvAccess(trial) = 0;
  }
  send_msg();
}

void do_work(long start, long end, void *result) {
  long tmp=0;
  for (long i=start; i<=end; i++) {
    tmp+=(long)(sqrt(1+cos(i*1.57)));
  }
  *(long *)result = tmp + *(long *)result;
}


void bigmsg_handler(void *vmsg)
{
  int i, next;
  message msg = (message)vmsg;
  // if this is a receiving PE
  if (CmiMyPe() >= CmiNumPes() / 2) {
    CpvAccess(recv_count) = 1 + CpvAccess(recv_count);
    long sum = 0;
    long result = 0;
    double num_ints = (CpvAccess(msg_size) - CmiMsgHeaderSizeBytes) / sizeof(int);
    double exp_avg = (num_ints - 1) / 2;
    for (i = 0; i < num_ints; ++i) {
      sum += msg->payload[i];
      do_work(i,sum,&result);
    }
    if(result < 0) {
      CmiPrintf("Error! in computation");
    }
    double calced_avg = sum / num_ints;
    if (calced_avg != exp_avg) {
      CmiPrintf("Calculated average of %f does not match expected value of %f, exiting\n", calced_avg, exp_avg);
      message exit_msg = (message) CmiAlloc(CpvAccess(msg_size));
      CmiSetHandler(exit_msg, CpvAccess(stop_index));
      CmiSyncBroadcastAllAndFree(CpvAccess(msg_size), exit_msg);
      return;
    } 
    // else
    //   CmiPrintf("Calculation OK\n"); // DEBUG: Computation Check
    if(CpvAccess(recv_count) == MSG_COUNT) {
      CpvAccess(recv_count) = 0;
      
      CmiFree(msg);
      msg = (message)CmiAlloc(CpvAccess(msg_size));
      CmiSetHandler(msg, CpvAccess(ackmsg_index));
      CmiSyncSendAndFree(0, CpvAccess(msg_size), msg);
    } else
      CmiFree(msg);
  } else
    CmiPrintf("\nError: Only node-1 can be receiving node!!!!\n");
}

void pe0_ack_handler(void *vmsg)
{
  int pe;
  message msg = (message)vmsg;
   //Pe-0 receives all acks
  CpvAccess(ack_count) = 1 + CpvAccess(ack_count);

  // DEBUG: Computation Print Check
  //CmiPrintf("All %d messages of size %d on trial %d OK\n", MSG_COUNT, CpvAccess(msg_size), CpvAccess(trial));
    

  if(CpvAccess(ack_count) == CmiNumPes()/2) {
    CpvAccess(ack_count) = 0;
    CpvAccess(total_time) = CmiWallTimer() - CpvAccess(total_time);

    // DEBUG: Original Print Statement
    //CmiPrintf("Received [Trial=%d, msg size=%d] ack on PE-#%d send time=%lf, process time=%lf, total time=%lf\n",
    //         CpvAccess(trial), CpvAccess(msg_size), CmiMyPe(), CpvAccess(send_time), CpvAccess(process_time), CpvAccess(total_time));

    CmiFree(msg);

    // store times in arrays
    send_time[CpvAccess(trial)] =  CpvAccess(send_time) * 1000000.0;       // convert to microsecs.
    process_time[CpvAccess(trial)] = CpvAccess(process_time) * 1000000.0;
    total_time[CpvAccess(trial)] = CpvAccess(total_time) * 1000000.0;

    CpvAccess(trial) = CpvAccess(trial) + 1;

    // print results
    if (CpvAccess(warmup_flag) || CpvAccess(trial) == nTRIALS_PER_SIZE) print_results();

    // if this is not the warmup round, and we have finished the final trial, and we are on the final msg size, exit
    if(!CpvAccess(warmup_flag) && CpvAccess(trial) == nTRIALS_PER_SIZE && CpvAccess(round) == nMSG_SIZE - 1)
    {
      message exit_msg = (message) CmiAlloc(CpvAccess(msg_size));
      CmiSetHandler(exit_msg, CpvAccess(stop_index));
      CmiSyncBroadcastAllAndFree(CpvAccess(msg_size), exit_msg);
      return;
    }
    else {
      // CmiPrintf("\nSending short msgs from PE-%d", CmiMyPe());
      for(pe = 0 ; pe<CmiNumPes() / 2; pe++) {
        int smsg_size = 4+CmiMsgHeaderSizeBytes;
        message smsg = (message)CmiAlloc(smsg_size);
        CmiSetHandler(smsg, CpvAccess(shortmsg_index));
        CmiSyncSendAndFree(pe, smsg_size, smsg);
      }
    }
  }
}


void bigmsg_init()
{
  int totalpes = CmiNumPes(); //p=num_pes
  int pes_per_node = totalpes/2; //q=p/2
  if (CmiNumPes()%2 !=0) {
    CmiPrintf("note: this test requires at multiple of 2 pes, skipping test.\n");
    CmiPrintf("exiting.\n");
    //CsdExitScheduler();
    message exit_msg = (message) CmiAlloc(CpvAccess(msg_size));
    CmiSetHandler(exit_msg, CpvAccess(stop_index));
    CmiSyncBroadcastAllAndFree(CpvAccess(msg_size), exit_msg);
    return;
  } else {
    if(CmiMyPe() < pes_per_node)
      send_msg();
  }
}



void bigmsg_moduleinit(int argc, char **argv)
{
  CpvInitialize(int, bigmsg_index);
  CpvInitialize(int, ackmsg_index);
  CpvInitialize(int, shortmsg_index);
  CpvInitialize(int, msg_size);
  CpvInitialize(int, trial);
  CpvInitialize(int, round);
  CpvInitialize(int, warmup_flag);
  CpvInitialize(int, recv_count);
  CpvInitialize(int, ack_count);
  CpvInitialize(double, total_time);
  CpvInitialize(double, send_time);
  CpvInitialize(double, process_time);
  CpvInitialize(int, stop_index);

  CpvAccess(bigmsg_index) = CmiRegisterHandler(bigmsg_handler);
  CpvAccess(shortmsg_index) = CmiRegisterHandler(shortmsg_handler);
  CpvAccess(ackmsg_index) = CmiRegisterHandler(pe0_ack_handler);
  CpvAccess(stop_index) = CmiRegisterHandler(stop);
  CpvAccess(msg_size) = 16+CmiMsgHeaderSizeBytes;
  CpvAccess(trial) = 0;
  CpvAccess(round) = 0;
  CpvAccess(warmup_flag) = 1;
  // Set runtime cpuaffinity
  CmiInitCPUAffinity(argv);
  // Initialize CPU topology
  CmiInitCPUTopology(argv);
  // Wait for all PEs of the node to complete topology init
        CmiNodeAllBarrier();

  // Update the argc after runtime parameters are extracted out
  argc = CmiGetArgc(argv);
  if(CmiMyPe() < CmiNumPes()/2)
    bigmsg_init();
}

int main(int argc, char **argv)
{
	ConverseInit(argc,argv,bigmsg_moduleinit,0,0);
}