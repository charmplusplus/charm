#include <stdlib.h>
#include "ccs-client.h"
#include "ccs-client.c"
#include <string.h>

int main (int argc, char** argv) {

  if (argc<3) return 1;

  char code[1000] = "ck.printstr('python')\n"
    "value=charm.runhigh()\n"
    "ck.printstr('python value: '+repr(value))\n";

  CcsServer server;
  char *host=argv[1];
  int port=atoi(argv[2]);

  CcsConnect (&server, host, port, NULL);

  // if there is a third argument means kill the server
  if (argc>3) CcsSendRequest (&server, "kill", 0, 1, code);
  else CcsSendRequest (&server, "pyCode", 0, strlen(code)+1, code);
}
