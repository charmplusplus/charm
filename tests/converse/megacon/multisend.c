#include <stdio.h>
#include <converse.h>

void Cpm_megacon_ack();

typedef struct
{
  char core[CmiMsgHeaderSizeBytes]; 
  int hdlrnum;
  double data[10];
} multisendmsg;

CpvDeclare(int, multisend_first_index);
CpvDeclare(int, multisend_second_index);
CpvDeclare(int, multisend_third_index);
CpvDeclare(int, multisend_recv_index);

CpvDeclare(int, multisend_replies);

void multisend_fail()
{
  CmiPrintf("data corrupted in multisend.\n");
  exit(1);
}

void multisend_first_handler(multisendmsg *msg)
{
  int i;

  if(msg->hdlrnum != 1)
    multisend_fail();
  for(i=0;i<10;i++) {
    if(msg->data[i] != (double) (i+1))
      multisend_fail();
  }
  CmiGrabBuffer((void **) &msg);
  CmiSetHandler(msg, CpvAccess(multisend_recv_index));
  CmiSyncSendAndFree(0, sizeof(multisendmsg), msg);
}

void multisend_second_handler(multisendmsg *msg)
{
  int i;

  if(msg->hdlrnum != 2)
    multisend_fail();
  for(i=0;i<10;i++) {
    if(msg->data[i] != (double) (i+2))
      multisend_fail();
  }
  CmiGrabBuffer((void **) &msg);
  CmiSetHandler(msg, CpvAccess(multisend_recv_index));
  CmiSyncSendAndFree(0, sizeof(multisendmsg), msg);
}

void multisend_third_handler(multisendmsg *msg)
{
  int i;

  if(msg->hdlrnum != 3)
    multisend_fail();
  for(i=0;i<10;i++) {
    if(msg->data[i] != (double) (i+3))
      multisend_fail();
  }
  CmiSetHandler(msg, CpvAccess(multisend_recv_index));
  CmiSyncSend(0, sizeof(multisendmsg), msg);
}

void multisend_recv_handler(multisendmsg *msg)
{
  int i;

  CpvAccess(multisend_replies)++;
  for(i=0;i<10;i++) {
    if(msg->data[i] != (double) (i+msg->hdlrnum))
      multisend_fail();
  }
  if(CpvAccess(multisend_replies)==3)
    Cpm_megacon_ack(CpmSend(0));
}

void multisend_init(void)
{
  multisendmsg first, third;
  multisendmsg *second;
  int sizes[3];
  char *msgs[3];
  int i;

  if(CmiNumPes() < 2) {
    CmiPrintf("Multisend requires at least 2 processors. skipping...\n");
    Cpm_megacon_ack(CpmSend(0));
  } else {
    second = (multisendmsg *) CmiAlloc(sizeof(multisendmsg));
    first.hdlrnum = 1;
    second->hdlrnum = 2;
    third.hdlrnum = 3;
    for(i=0;i<10;i++) {
      first.data[i] = (double) (i+1);
      second->data[i] = (double) (i+2);
      third.data[i] = (double) (i+3);
    }
    CmiSetHandler(&first, CpvAccess(multisend_first_index));
    CmiSetHandler(second, CpvAccess(multisend_second_index));
    CmiSetHandler(&third, CpvAccess(multisend_third_index));
    sizes[0] = sizes[1] = sizes[2] = sizeof(multisendmsg);
    msgs[0] = (char *) &first; 
    msgs[1] = (char *) second; 
    msgs[2] = (char *) &third;
    CmiMultipleSend(1, 3, sizes, msgs);
    CmiFree(second);
  }
  CpvAccess(multisend_replies) = 0;
}

void multisend_moduleinit()
{
  CpvInitialize(int, multisend_first_index);
  CpvInitialize(int, multisend_second_index);
  CpvInitialize(int, multisend_third_index);
  CpvInitialize(int, multisend_recv_index);
  CpvInitialize(int, multisend_replies);
  CpvAccess(multisend_first_index) = 
    CmiRegisterHandler(multisend_first_handler);
  CpvAccess(multisend_second_index) = 
    CmiRegisterHandler(multisend_second_handler);
  CpvAccess(multisend_third_index) = 
    CmiRegisterHandler(multisend_third_handler);
  CpvAccess(multisend_recv_index) = 
    CmiRegisterHandler(multisend_recv_handler);
}



