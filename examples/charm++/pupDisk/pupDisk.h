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

class pupDiskMap: public CkArrayMap
{
 public:
  int maxFiles;
 pupDiskMap(int _maxFiles):maxFiles(_maxFiles) {}
  inline int procNum(int, const CkArrayIndex &iIndex)
  {
    int *index=(int *) iIndex.data();
    int proc;
    if(CmiCpuTopologyEnabled())
      { // use physnode API
	if(CmiNumPhysicalNodes() > maxFiles)
	  {
	    proc=CmiGetFirstPeOnPhysicalNode(index[0]);
	  }
	else
	  { 
	    //cleverness could be tried, but we really don't care because you 
	    //want more files than is good for you.
	    proc=index[0]%CmiNumPes();
	  }
      }
    else
      {
	if(CmiNumNodes()>maxFiles)
	  {// 
	    proc=index[0]*CmiMyNodeSize();
	  }
	else if (CmiNumPes()>maxFiles)
	  { //simple round robin because we don't really care
	    proc=index[0];
	  }
	else //there is no good mapping
	  {
	    proc=index[0]%CkNumPes();
	  }
      }
    return proc;
  }
  
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
