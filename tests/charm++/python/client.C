#include <stdlib.h>
#include "ccs-client.h"
#include "PythonCCS-client.h"
#include <string.h>
#include <string>
#include <iostream>
#include <sys/time.h>
#include <time.h>


double wallTimer(){
  struct timeval tv;

  gettimeofday(&tv, NULL);	
  return ((double )tv.tv_sec + ((double )tv.tv_usec)*1e-6);
}

class MyIterator : public PythonIterator {
public:
  int s;
  MyIterator(int i) : s(i) { }

  int size() {return 4;}
  char *pack() {
    void *memory = malloc(4);
    *(int*)memory = s;
    return (char *)memory;

  }
};

int main (int argc, char** argv) {

  if (argc<5) return 1;

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
  int iter=atoi(argv[3]);
  int reuse = 0;
  reuse = atoi(argv[4]);
  if(reuse > 0)
   reuse = 1;
  int useGroups = atoi(argv[5]);

  CcsConnect (&server, host, port, NULL);

  PythonExecute *wrapper;
  char *pythonString;

  if (useGroups > 0) {
    MyIterator *myIter = new MyIterator(htonl(4));
    printf("size: %d (%zu)\n",myIter->size(),sizeof(*myIter));
    static char s_metodo[] = "metodo";
    wrapper = new PythonExecute(const_cast<char*>(code.c_str()), s_metodo, myIter);
    static char s_CpdPythonGroup[] = "CpdPythonGroup";
    pythonString = s_CpdPythonGroup;
  } else {
    wrapper = new PythonExecute(const_cast<char*>(code.c_str()));
    static char s_pyCode[] = "pyCode";
    pythonString = s_pyCode;
  }

  wrapper->setHighLevel(true);
  wrapper->setKeepPrint(true);
  if(reuse < 0){
    wrapper->setPersistent(false);
  }else{
    wrapper->setPersistent(true);
  }	  
  wrapper->setWait(true);

  // if there is a third argument means kill the server
  CmiUInt4 remoteValue;
  CmiUInt4 interpreter;
  char buffer[100];
  double _startTime;


  if(reuse>0){
     _startTime = wallTimer();

     CcsSendRequest (&server, pythonString, 0, wrapper->size(), wrapper->pack());
     CcsRecvResponse (&server, 10, &interpreter, sizeof(interpreter));
     double duration = wallTimer() - _startTime;

     printf("buffer: %d   duration %.6lf \n",interpreter,duration);

     PythonPrint request(interpreter);
 //    sleep(2);
      //request.print();
      CcsSendRequest (&server, pythonString, 0, request.size(), request.pack());
      //request.print();
      CcsRecvResponse (&server, 100, buffer, 100);
      //request.print();
      printf("response: %s\n",buffer);
  	
  }
  for(int i=0;i<iter;i++){
     _startTime = wallTimer();
     if(reuse>0){
       wrapper->setInterpreter(interpreter);
       //char *tmp = " ";
       //wrapper->setCode(tmp);
     }  
     CcsSendRequest (&server, pythonString, 0, wrapper->size(), wrapper->pack());
     CcsRecvResponse (&server, 10, &remoteValue, sizeof(remoteValue));
     double duration = wallTimer() - _startTime;
     

     printf("buffer: %d   duration %.6lf \n",remoteValue,duration);
     PythonPrint request(remoteValue);
      //request.print();
     CcsSendRequest (&server, pythonString, 0, request.size(), request.pack());
      //request.print();
     CcsRecvResponse (&server, 100, buffer, 100);
      //request.print();
     printf("response: %s\n",buffer);
     sleep(2);
      //sleep(2);
      //request.print();
      //    CcsSendRequest (&server, "pyCode", 0, sizeof(request), &request);
      //    CcsRecvResponse (&server, 100, buffer, 100);
      //printf("responce: %x\n",*buffer);
      //
  }
  CcsSendRequest (&server, "kill", 0, 1, code.c_str());
}
