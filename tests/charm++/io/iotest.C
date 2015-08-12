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
  char *ReadArray; // ReadArray pointer contains the pointer for the read buffer
  char fileName[50];
  int fileSize;
  int remainingBytes;
  int fixedBytesToRead;
 
  std::vector<Ck::IO::File> f;  
public:
  Main(CkArgMsg *m) {
    numdone = 0;
    n = atoi(m->argv[1]);

    sprintf(fileName,"%s", m->argv[m->argc - 1]);

    FILE *fp = fopen(fileName, "r");

    if(fp == NULL)
      CkPrintf("Unable to open the file \n");
    
    fseek(fp, 0L, SEEK_END); // This iterate through the file 
    
    fileSize = ftell(fp);     //This will give you the file size

    fclose(fp);

    fixedBytesToRead = floor(fileSize / n);
    remainingBytes = fileSize % n;
    ReadArray = new char[fileSize];

    CkPrintf("Numchares = %d Bytes %d \n", atoi(m->argv[m->argc - 2]), fileSize);
    
    f.resize(1);      
    
    for (int i = 0; i < f.size(); ++i){
      thisProxy.run(4*i);
    }

    mainProxy= thisProxy;

    CkPrintf("Main ran\n");
    delete m;

  }

  void iterDone(CkReductionMsg *m) { // checks all the files  are done
    numdone++;
    
    if (numdone == f.size()){
      
   
      int fd = open(fileName, O_RDONLY);
      if(fd  < 0)
        CkPrintf("Cannot Open File \n");

      char ReadDataFromFile[fileSize];

      size_t readData = pread(fd, ReadDataFromFile, fileSize, 0);

      for(int i = 0; i < readData; i++){
        //CkPrintf ("%d %d %d %d\n", ReadArray[i], ReadDataFromFile[i], readData, numdone);
        if(ReadArray[i] != ReadDataFromFile[i]){
          CkPrintf("ReadData is not Valid \n");  
          CkExit(); 
        }
      }

      CkPrintf("ReadData is Corrrect \n");
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
  int fixedReadBytes;
  int remainingReadBytes;

  
  int issuedRead = 0;

  test(Ck::IO::Session token,int n, int fixedBytesToRead, int remainingBytes) { // Pass the number of chare elements in our case they are n elements
    
    readData = new char[fixedBytesToRead];
    int fd;
    int i = 0;
    int bytes = 20;

    fixedReadBytes = fixedBytesToRead;
    remainingReadBytes = remainingBytes;


    CkCallback myCB(CkIndex_test::readCompleted(NULL), thisProxy[thisIndex]);
     
    if (remainingReadBytes > 0 && thisIndex == 0){
      
      remainingReadBytes = remainingReadBytes + fixedReadBytes;
      Ck::IO::read(token, readData, remainingReadBytes, remainingReadBytes*thisIndex, myCB);
     
      
    }else{
    
      Ck::IO::read(token, readData, fixedReadBytes, fixedReadBytes*thisIndex + remainingReadBytes, myCB);
    
    }

    
    savedToken = token;
    numChares = n;
 
  }
  
  test(CkMigrateMessage *m) {
    
  }

  void readCompleted(Ck::IO::SessionReadyMsg* m) {

    if (remainingReadBytes > 0 && thisIndex == 0)
      mainProxy.testReadFiles(remainingReadBytes*thisIndex, remainingReadBytes,readData);
    else
      mainProxy.testReadFiles(fixedReadBytes*thisIndex + remainingReadBytes, fixedReadBytes,readData);

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