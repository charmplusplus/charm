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
  fscanf(fp,"CHANNELBW %d\n",&switchC_BW);
  fscanf(fp,"CHANNELDELAY %d\n",&switchC_Delay);
  fscanf(fp,"COLLECTION_INTERVAL %d\n",&collection_interval);
  fscanf(fp,"DISPLAY_LINK_STATS %d\n",&linkstats_on);
  fscanf(fp,"DISPLAY_MESSAGE_DELAY %d\n",&msgstats_on);
  fclose(fp);
}
