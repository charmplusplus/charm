#include "iotest.decl.h"
#include <vector>

#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <pup_stl.h>
#include <unistd.h>

#include <string>
#include <map>
#include <algorithm>
#include <sstream>
#include <sys/types.h>

CProxy_Main mainProxy;

class Main : public CBase_Main {
  Main_SDAG_CODE

  CProxy_test testers;
  int n, numdone;
  char ReadArray[100*100];
  std::vector<Ck::IO::File> f;  
public:
  Main(CkArgMsg *m) {
    numdone = 0;
    n = atoi(m->argv[1]);

    f.resize(1);      
    for (int i = 0; i < f.size(); ++i)
      thisProxy.run(4*i);

    mainProxy= thisProxy;

    CkPrintf("Main ran\n");
    delete m;

  }

  void iterDone(CkReductionMsg *m) { // checks all the files  are done
    numdone++;
    
    if (numdone == f.size()){
      
   
      int fd = open("test0", O_RDONLY);
      if(fd  < 0)
        CkPrintf("Cannot Open File \n");

      char ReadDataFromFile[100*100];

      size_t readData = pread(fd, ReadDataFromFile, 90, 0);

      for(int i = 0; i < readData; i++)
        CkPrintf ("%d %d %d %d\n", ReadArray[i], ReadDataFromFile[i], readData, numdone);

      if(strcmp(ReadArray,ReadDataFromFile)==0)
        CkPrintf("ReadData is Corrrect \n");
      else {
        CkPrintf("ReadData is not Valid \n");
      } 

      CkExit();

    }
  }

  void testReadFiles(int offset, int size, char data[]){
  
    
      memcpy(ReadArray + offset, data,size);

 
  } 

};

struct test : public CBase_test {
  int numDone=0;
  int instance = 0;
  Ck::IO::Session savedToken;
  int numChares;
  char *readData;
  
  test(Ck::IO::Session token,int n) { // Pass the number of chare elements in our case they are n elements
    
    readData = new char[20];
    int fd;
    int i = 0;
    int bytes = 20;
    
    CkCallback myCB(CkIndex_test::readCompleted(NULL), thisProxy[thisIndex]);
     
    Ck::IO::read(token, readData, 10, 10*thisIndex, myCB);

    savedToken = token;
    numChares = n;
 
  }
  
  test(CkMigrateMessage *m) {
    
  }

  void readCompleted(Ck::IO::SessionReadyMsg* m) {

    mainProxy.testReadFiles(10*thisIndex,10,readData);

    CkPrintf("On %d (0x%x): %d %d %d %d %d %d %d %d %d %d\n", thisIndex,readData, readData[0],readData[1],readData[2],readData[3],readData[4],readData[5],readData[6],readData[7],readData[8],readData[9]);  
    thisProxy[0].done(savedToken,numChares);   
  }

  void done(Ck::IO::Session token, int n) { // This is done function
      
    numDone++;
      if(numDone == n){ // When numDone equals to the total number of Pe close
        Ck::IO::endSession(token);   // Create End Session in the Library and End the Session
      }
  }
};

#include "iotest.def.h"