#include "util.h"

void Config::readConfig(CkArgMsg *m) {
  
  FILE *fp;
  fp = fopen("netconfig","r");
  
  if(fp == NULL) {
    CkPrintf(" Could not open configuration file \"netconfig\" \n");
    CmiAbort(" Please Make sure it's in $PATH \n");
  }
  
  
  if(m->argc < 2) {
    CkPrintf(" Usage pgm <netsim_status> \n");
    CmiAbort(" Too few arguments  \n");
  }

  netsim_on = atoi(m->argv[1]);
  skip_on = atoi(m->argv[2]);
  
  fscanf(fp,"MAX_PACKET_SIZE %d\n",&maxpacksize);
  fscanf(fp,"SWITCH_VC %d\n",&switchVc);
  fscanf(fp,"SWITCH_PORT %d\n",&numP);  // Arrgh needed this for indirect networks ,  8 for quarternary fat-tree and so on
  fscanf(fp,"SWITCH_BUF %d\n",&switchBufsize);
  fscanf(fp,"CHANNELBW %f\n",&switchC_BW);
  fscanf(fp,"CHANNELDELAY %d\n",&switchC_Delay);
  fscanf(fp,"COLLECTION_INTERVAL %d\n",&collection_interval);
  fscanf(fp,"DISPLAY_LINK_STATS %d\n",&linkstats_on);
  fscanf(fp,"DISPLAY_MESSAGE_DELAY %d\n",&msgstats_on);
  fscanf(fp,"RECEPTION_SERIAL %d\n",&receptionSerial);
  fscanf(fp,"INPUT_SPEEDUP %d\n",&inputSpeedup);
/*
  fscanf(fp, "HEADER_SIZE %d\n", &headerSize);
  fscanf(fp, "CREDIT_SIZE %d\n", &creditSize);
  fscanf(fp, "NUM_NODES %d\n", &numNodes);

  fscanf(fp, "VC_BUF_CREDITS %d\n", &VCBufSize);
  fscanf(fp, "HCA_BUF_CREDITS %d\n", &HCABufSize);

  fscanf(fp, "DMA_DELAY_SMALL %d\n", &DMADelaySmall);
  fscanf(fp, "DMA_DELAY_LARGE %d\n", &DMADelayLarge);

  fscanf(fp, "SWITCH_DELAY %d\n", &switchDelay);
  fscanf(fp, "NUM_RAILS %d\n", &numRails);
 
  fscanf(fp, "HCA_SEND_DELAY %d\n", &HCASendDelay);
  fscanf(fp, "HCA_RECV_DELAY %d\n", &HCARecvDelay);

  fscanf(fp, "PROC_SEND_DELAY %d\n", &procSendOverhead);
  fscanf(fp, "PROC_RECV_DELAY %d\n", &procRecvOverhead);

  fscanf(fp, "HCA_SEND_PKT_DELAY %d\n", &HCAPktSendDelay);
  fscanf(fp, "HCA_RECV_PKT_DELAY %d\n", &HCAPktRecvDelay);

  //Messages smaller than this can be injected into the cache.
  //Simplifying assumption for not simulating the caches
  fscanf(fp, "INJECTION_THRESHOLD %d \n", &cacheInjectionThreshold);

  adaptiveRouting = 0;
  fscanf(fp, "ADAPTIVE_ROUTING %d\n", &adaptiveRouting);
*/
  fclose(fp);
}
