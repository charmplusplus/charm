#include <stdlib.h>
#include "ccs-client.h"
#include "ccs-client.c"
#include <string.h>
#include <string>
#include <iostream>

int main (int argc, char** argv) {

  if (argc<3) return 1;

  char codeline[1000];
  std::string code;
  int codelinesize;
  while ((codelinesize=read(0, codeline, 1000))>0) {
    code += std::string(codeline,codelinesize);
  }

  std::cout << "code: {\n"<< code << "}\n";

  CcsServer server;
  char *host=argv[1];
  int port=atoi(argv[2]);

  CcsConnect (&server, host, port, NULL);

  // if there is a third argument means kill the server
  char buffer[10];
  if (argc>3) {
    CcsSendRequest (&server, "kill", 0, 1, code.c_str());
  }
  else {
    CcsSendRequest (&server, "pyCode", 0, code.length()+1, code.c_str());
    CcsRecvResponse (&server, 10, buffer, 100);
    printf("buffer: %d\n",*(int*)buffer);
  }
}
