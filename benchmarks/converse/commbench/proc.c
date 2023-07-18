#include <converse.h>
#include <stdlib.h>
#include <math.h>
#include "commbench.h"

#define NTRIALS 4000000

CpvStaticDeclare(int, success);
CpvStaticDeclare(int, nreported);
CpvStaticDeclare(int, isSingle);
CpvStaticDeclare(double, Time1);
CpvStaticDeclare(double, TimeN);
CpvStaticDeclare(int, trial_handler);
CpvStaticDeclare(int, collect_handler);
CpvStaticDeclare(double, seqPI);
CpvStaticDeclare(double, parPI);

typedef struct pmsg {
  char core[CmiMsgHeaderSizeBytes];
  int success;
} ProcMsg;

static void doTrials(ProcMsg* msg) {
  int i, success = 0;
  int total = msg->success;
  double x, y;

  CrnSrand(0);
  for (i = 0; i < total; i++) {
    x = CrnDrand() - 0.5;
    y = CrnDrand() - 0.5;

    if ((x * x + y * y) <= 0.25) success++;
  }
  msg->success = success;
  CmiSetHandler(msg, CpvAccess(collect_handler));
  CmiSyncSendAndFree(0, sizeof(ProcMsg), msg);
}

static int iround(double x) { return (int)(ceil(2.0 * x) / 2.0); }

static void collectNumbers(ProcMsg* msg) {
  int npes;
  EmptyMsg emsg;
  CmiInitMsgHeader(emsg.core, sizeof(EmptyMsg));

  if (CpvAccess(isSingle)) {
    CpvAccess(Time1) = CmiWallTimer() - CpvAccess(Time1);
    CpvAccess(seqPI) = 4.0 * msg->success / NTRIALS;
    CpvAccess(isSingle) = 0;
    CpvAccess(nreported) = 0;
    CpvAccess(success) = 0;
    msg->success = NTRIALS / CmiNumPes();
    CmiSetHandler(msg, CpvAccess(trial_handler));
    CmiSyncBroadcastAll(sizeof(ProcMsg), msg);
    CpvAccess(TimeN) = CmiWallTimer();
    printf("if\n");
  } else {
    printf("else\n");
    CpvAccess(nreported)++;
    CpvAccess(success) += msg->success;
    if (CpvAccess(nreported) == CmiNumPes()) {
      CpvAccess(TimeN) = CmiWallTimer() - CpvAccess(TimeN);
      CpvAccess(parPI) = 4.0 * CpvAccess(success) / NTRIALS;
      npes = iround(CpvAccess(Time1) / CpvAccess(TimeN));
      CmiPrintf("[proc] Tseq = %le seconds, Tpar = %le seconds\n",
          CpvAccess(Time1), CpvAccess(TimeN));
      CmiPrintf("[proc] CmiNumPes() reported %d processors\n", CmiNumPes());
      CmiPrintf("[proc] But actual number of processors is %d\n", npes);
      CmiPrintf("[proc] FYI, appox PI (seq) = %lf\n", CpvAccess(seqPI));
      CmiPrintf("[proc] FYI, appox PI (par) = %lf\n", CpvAccess(parPI));
      CmiSetHandler(&emsg, CpvAccess(ack_handler));
      CmiSyncSend(0, sizeof(EmptyMsg), &emsg);
      printf("else if\n");
    }
  }
}

void proc_init(void) {
  ProcMsg msg;
  CmiInitMsgHeader(msg.core, sizeof(EmptyMsg));

  CpvAccess(isSingle) = 1;
  msg.success = NTRIALS;
  CmiSetHandler(&msg, CpvAccess(trial_handler));
  CmiSyncSend(0, sizeof(ProcMsg), &msg);
  CpvAccess(Time1) = CmiWallTimer();
  printf("proc init\n");
}

void proc_moduleinit(void) {
  CpvInitialize(int, success);
  CpvInitialize(int, nreported);
  CpvInitialize(int, collect_handler);
  CpvInitialize(int, trial_handler);
  CpvInitialize(double, Time1);
  CpvInitialize(double, TimeN);
  CpvInitialize(int, isSingle);
  CpvInitialize(double, seqPI);
  CpvInitialize(double, parPI);

  CpvAccess(collect_handler) = CmiRegisterHandler((CmiHandler)collectNumbers);
  CpvAccess(trial_handler) = CmiRegisterHandler((CmiHandler)doTrials);
  printf("proc module init\n");
}
