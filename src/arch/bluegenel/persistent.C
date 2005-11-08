
#if CMK_PERSISTENT_COMM

extern "C" void setupRecvSlot(PersistentReceivesTable *slot, int maxbytes) {
  
  int i;
  for (i = 0; i<PERSIST_BUFFERS_NUM; i++) {
    char *buf = (char *)CmiAlloc((maxbytes + sizeof(BLRMA_Put_Recv_t) + 16));
    BLRMA_Put_Recv_t *recv = (BLRMA_Put_Recv_t *) ALIGN_16(buf + maxbytes);
    
    BLRMA_Put_Recv_Init(recv, buf, maxbytes, persist_recv_done, buf);
    
    //We return recv pointer because that will tell the sender where the 
    //persistent data structure is
    slot->messagePtr[i] = recv;
    slot->recvSizePtr[i] = NULL;
  }
}

extern "C" void *PerAlloc(int size) {
  return CmiAlloc(size);
}

//Looks like a really bad hack !!
extern "C" void PerFree(char *buf) {

  char *pbuf =  (char *)((unsigned long *)buf)[0];
  CmiFree(pbuf);
}

extern "C" int machineSendPersistentMsg(SMSG_LIST *msg_tmp) {

  //printf("machine persistent send\n");

  PersistentSendsTable *slot = NULL;

  CmiAssert(msg_tmp->phscount < msg_tmp->phsSize);

  if(msg_tmp->phsSize == 1) 
    slot =  (PersistentSendsTable *)(msg_tmp->phs);
  else {
    slot = ((PersistentSendsTable **)msg_tmp->phs)[msg_tmp->phscount];
  }
  
  BLRMA_Put_Send_t *send_buf;

  CmiAssert(slot->used == 1);
  CmiAssert(slot->destPE == msg_tmp->destpe);

  if (msg_tmp->size > slot->sizeMax) {
    CmiPrintf("size: %d sizeMax: %d\n", msg_tmp->size, slot->sizeMax);
    CmiAbort("Abort: Invalid size\n");
  }

  if (slot->destAddress[0]) {
    msg_tmp->send_buf = (char *)malloc (sizeof(BLRMA_Put_Send_t));
    send_buf = (BLRMA_Put_Send_t *) msg_tmp->send_buf;
    BLRMA_Put ( send_buf, _msgr, msg_tmp->destpe, msg_tmp->msg, msg_tmp->size, 
		(BLRMA_Put_Recv_t*) slot->destAddress[0], send_done, msg_tmp);
    return 1;
  }
  else {
    return 0;
  }
  return 0;
}

extern "C" void CmiSendPersistentMsg(PersistentHandle h, int destpe, int size, void *m) {
  CmiUsePersistentHandle(&h, 1);
  CmiSyncSendAndFree(destpe, size, m);
}

extern "C" void persist_machine_init() {
}

#endif
