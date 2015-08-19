#include "ccs-client.h"
//#include "ccs-client.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void  usage(void)
{
  fprintf(stderr, "Usage: client svrIP svrPort\n");
  exit(1);
}

int main(int argc, char **argv)
{
  CcsServer svr;
  CcsSec_secretKey *key=NULL,keySto;
  int i;
  char *sendWhat="milind";
  char reply[1024];
  if(argc < 3) {
    usage();
  }
  if(argc>3 && CCS_AUTH_makeSecretKey(argv[3],&keySto))
	   key=&keySto;
  
  CcsConnect(&svr, argv[1], atoi(argv[2]), key);
  for(i=0;i<CcsNumPes(&svr);i++) {
    CcsSendRequest(&svr, "ping", i, strlen(sendWhat)+1, sendWhat);
    CcsRecvResponse(&svr, 1023, reply,60);
    printf("Reply: %s", reply);
  }

  for(i=0;i<CcsNumPes(&svr);i++) {
    CcsSendRequest(&svr, "ping", i, strlen(sendWhat)+1, sendWhat);
    CcsRecvResponse(&svr, 1023, reply,60);
    printf("Reply: %s", reply);
  }
  exit(0);
}

