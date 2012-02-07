//////////////////////////////////////
//
//  pupDisk.h  
//
//  Declaration of chares in pupDisk
//
//  Author: Eric Bohm
//  Date: 2012/01/23
//
//////////////////////////////////////

#include "pupDisk.decl.h"
#include <map>
#include "pup_stl.h"
class main : public CBase_main {
public:
  main(CkMigrateMessage *m) {}
  main(CkArgMsg *m);
  void done(CkReductionMsg *m){
    CkPrintf("done\n");
    CkExit();
  }
  void initialized(CkReductionMsg *m);
  void written(CkReductionMsg *m);
  void read(CkReductionMsg *m);

};


class userData : public CBase_userData {
public:
  userData(CkMigrateMessage *m) {}
 userData(int _howbig, int _numElements, int _maxFiles): howBig(_howbig), numElements(_numElements), maxFiles(_maxFiles){ myData=new someData(howBig);}
  ~userData(){ if(myData) delete myData;}
  void init();
  void read();
  void write();
  void writeDone();
  void verify();
  void acceptData(someData &inData);
 private:
  someData *myData;
  int howBig;
  int numElements;
  int maxFiles;
};

class pupDisk : public CBase_pupDisk {
public:
  pupDisk(CkMigrateMessage *m) {}
  pupDisk(int _howbig, int _numElements, int _maxFiles);
  ~pupDisk(){ ;}
  void read(int sender);
  void write(int sender, someData &data);
  void diskWrite();
  bool diskRead(int sender);
 private:
  someData *dataCache;
  bool doneRead;
  int count;
  int howBig;
  int numElements;
  int maxFiles;
  int elementsToWrite;
  std::map<int, int> lookupIdx;
  int nextSlot;
};
