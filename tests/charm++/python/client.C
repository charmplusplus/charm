#include <stdlib.h>
#include "ccs-client.h"
#include "ccs-client.c"
#include "PythonCCS-client.h"
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

  PythonExecute wrapper((char*)code.c_str());
  wrapper.setHighLevel(true);
  wrapper.setKeepPrint(true);
  wrapper.setPersistent(true);

  // if there is a third argument means kill the server
  CmiUInt4 remoteValue;
  char buffer[100];
  if (argc>3) {
    CcsSendRequest (&server, "kill", 0, 1, code.c_str());
  }
  else {
    CcsSendRequest (&server, "pyCode", 0, wrapper.size(), wrapper.pack());
    CcsRecvResponse (&server, 10, &remoteValue, sizeof(remoteValue));
    printf("buffer: %d\n",remoteValue);
    PythonPrint request(remoteValue);
    sleep(2);
    //request.print();
    //   CcsSendRequest (&server, "pyCode", 0, sizeof(request), &request);
    //request.print();
    //   CcsRecvResponse (&server, 100, buffer, 100);
    //request.print();
    printf("responce: %s\n",buffer);
    //sleep(2);
    //request.print();
    //   CcsSendRequest (&server, "pyCode", 0, sizeof(request), &request);
    //   CcsRecvResponse (&server, 100, buffer, 100);
    printf("responce: %x\n",*buffer);
  }
}
