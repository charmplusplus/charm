////////////////////////////////////
//
//  pupDisk.C
//
//  Definition of chares in pupDisk
//
//  Author: Eric Bohm
//  Date: 2012/01/23
//
////////////////////////////////////

#include "pupDisk.h"
CkCallback icb, rcb, wcb, vcb;
CProxy_userData userDataProxy;
CProxy_pupDisk pupDiskProxy;
int numElementsPer;
main::main(CkArgMsg *m)
{

  int numElements=10;
  int size=20;
  bool skipToRead=false;
  int maxFiles=CkNumPes();
  if(CmiCpuTopologyEnabled())
    {
      maxFiles=CmiNumPhysicalNodes();
    }
  if(m->argc>1)
    numElements=atoi(m->argv[1]);
  if(m->argc>2)
    size=atoi(m->argv[2]);
  if(m->argc>3)
    maxFiles=atoi(m->argv[3]);
  if(m->argc>4)
    skipToRead=(m->argv[4][0]=='r');
  delete m;
  if(numElements/maxFiles<=0)
    CkAbort("This works better with more elements than files");
  //rejigger their choices, possibly reducing the number of files below max
  numElementsPer=numElements/maxFiles;
  if(numElements%maxFiles>0) ++numElementsPer;
  maxFiles=numElements/numElementsPer;
  if(numElements%numElementsPer) ++maxFiles;
  CkPrintf("pupDisk numElements %d howBig %d maxFiles %d skip %d elements per file %d\n", numElements, size, maxFiles, skipToRead, numElementsPer);
  icb = CkCallback(CkIndex_main::initialized(NULL),  thishandle);
  wcb = CkCallback(CkIndex_main::written(NULL),  thishandle);
  rcb = CkCallback(CkIndex_main::read(NULL),  thishandle);
  vcb = CkCallback(CkIndex_main::done(NULL),  thishandle);
  CProxy_pupDiskMap diskMapProxy = CProxy_pupDiskMap::ckNew(maxFiles);
  CkArrayOptions mapOptions(maxFiles);
  mapOptions.setMap(diskMapProxy);
  pupDiskProxy= CProxy_pupDisk::ckNew(size,numElements,maxFiles,mapOptions);
  pupDiskProxy.doneInserting();
  userDataProxy= CProxy_userData::ckNew(size,numElements,maxFiles, numElements);
  userDataProxy.doneInserting();
  if(skipToRead)
    {
      CkPrintf("reading data\n");
      userDataProxy.read();
    }
  else
    {
      userDataProxy.init();
    }
}



void main::initialized(CkReductionMsg *m)
  {
    CkPrintf("writing data\n");
    userDataProxy.write();
  }
void main::written(CkReductionMsg *m)
  {
    CkPrintf("reading data\n");
    userDataProxy.read();
  }
void main::read(CkReductionMsg *m)
  {
    CkPrintf("verifying data\n");
    userDataProxy.verify();
  }

void userData::init(){
  CkAssert(myData); 
  for (int i=0;i<howBig;++i) myData->data[i]=thisIndex;
  contribute(sizeof(int), &thisIndex, CkReduction::sum_int, icb);
}

void userData::verify(){
  CkAssert(myData); 
  for (int i=0;i<howBig;++i) 
    if(myData->data[i]!=thisIndex){
      CkPrintf("[%d] element %d corrupt as %d\n", 
	       thisIndex, i, myData->data[i]);
      CkAbort("corrupt element");
    }
  CkPrintf("[%d] verified\n",thisIndex);
  contribute(sizeof(int), &thisIndex, CkReduction::sum_int, vcb);
}

void userData::write()
{
  
  int fileNum = thisIndex/numElementsPer;
  //  CkPrintf("[%d] userData write to file %d\n",thisIndex,fileNum);
  pupDiskProxy[fileNum].write(thisIndex, *myData);
}

void userData::read()
{
  int fileNum = thisIndex/numElementsPer;
  pupDiskProxy[fileNum].read(thisIndex);
}

void userData::acceptData(someData &inData){
  for(int i=0; i<howBig; ++i) myData->data[i]=inData.data[i];
  contribute(sizeof(int), &thisIndex, CkReduction::sum_int, rcb);
}

pupDisk::pupDisk(int _howbig, int _numElements, int _maxFiles): howBig(_howbig), numElements(_numElements), maxFiles(_maxFiles)
  { elementsToWrite=numElementsPer; 
    if(thisIndex==maxFiles-1 && numElements%numElementsPer>0) elementsToWrite=numElements%numElementsPer; 
    dataCache=new someData[elementsToWrite]; 
    count=0; 
    nextSlot=0; 
    //    CkPrintf("[%d] pupDisk constructed expecting elementsToWrite %d for / %d and %% %d\n",thisIndex, elementsToWrite, numElements/maxFiles, numElements%maxFiles);
  }



void pupDisk::read(int sender)
{
  if(diskRead(sender))
    {
      // the ugly verbose syntax for extracting what you want from an STL map
      // never fails to annoy me.
      int offset=(*lookupIdx.find(sender)).second;
      userDataProxy[sender].acceptData(dataCache[offset]);
    }
}

bool pupDisk::diskRead(int sender)
{
  if(!doneRead)
    {
      // get stuff from disk

      // a more complicated caching scheme could pull less than the
      // entire file and use a per entry flag system to track what is
      // in cache.
      doneRead=true;      
      //      CkPrintf("[%d] reading from file for %d\n",thisIndex, sender);
      char *d = new char[512];
      sprintf(d, "%s.%d.%d.%d", "diskfile", numElements, howBig, thisIndex);
      FILE *f = fopen(d,"r");
      if (f == NULL) {
	CkPrintf("[%d] Open failed with %s. \n", CkMyPe(), d);
	CkAbort("\n");
      }
      // A simple scheme would require the user be consistent in their
      // parameter choices across executions.  A more elaborate scheme
      // codifies them in a block so the reader can do a lookup for
      // the parameters used during writing.
      PUP::fromDisk pd(f);
      PUP::machineInfo machInfor;
      pd((char *)&machInfor, sizeof(machInfor));       // machine info
      if (!machInfor.valid()) {
	CkPrintf("Invalid machineInfo on disk file when reading %d!\n", thisIndex);
	CkAbort("");
      }
      PUP::xlater p(machInfor, pd);
      int elementsToWriteFile;
      p|elementsToWriteFile;
      // safety check, for some formats you might be able to adjust
      // properly if the file's parameters disagree from your instance's.
      // This implementation is not that smart.
      if(elementsToWriteFile==elementsToWrite)
	{
	  p|lookupIdx;
	  someData input;
	  for(int i=0;i<elementsToWrite;++i)
	    {
	      dataCache[i].pup(p);
	    }
	}
      else
	{
	  CkAbort("a pox upon your file format");
	}
      fclose(f);
      delete [] d;
    }
  return doneRead;
}

void pupDisk::write(int sender, someData &inData)
{
  //  CkPrintf("[%d] pupDisk write for sender %d with count %d of elementsToWrite %d\n",thisIndex, sender, count, elementsToWrite);
  lookupIdx[sender]=nextSlot;
  dataCache[nextSlot++]=inData;
  if(++count==elementsToWrite) 
    diskWrite();

}

void pupDisk::diskWrite()
{
  //  CkPrintf("[%d] writing to file\n",thisIndex);
  char *d = new char[512];
  sprintf(d, "%s.%d.%d.%d", "diskfile", numElements, howBig, thisIndex);
  FILE *f;
  struct stat sb;
  // a+ will force appending, which is not what we want
  if(stat(d,&sb)==-1){
      f = fopen(d,"w");  
  }
  else
    {
      f = fopen(d,"r+");
    }
  if (f == NULL) {
    CkPrintf("[%d] Open for writing failed with %s \n", CkMyPe(), d);
    CkAbort("\n");
  }
  PUP::toDisk p(f);
  const PUP::machineInfo &machInfow = PUP::machineInfo::current();
  //  CkPrintf("[%d] writing machineInfo %d bytes\n",thisIndex,sizeof(machInfow));
  p((char *)&machInfow, sizeof(machInfow));       // machine info
  if(!machInfow.valid())
    {
      CkPrintf("Invalid machineInfo on disk file when writing %d!\n", thisIndex);
      CkAbort("");
    }
  p|elementsToWrite;
  p|lookupIdx;
  for(int i=0; i<elementsToWrite;i++)
    dataCache[i].pup(p);
  fflush(f);
  fclose(f);
  contribute(sizeof(int), &thisIndex, CkReduction::sum_int, wcb);
  delete [] d;
}



#include "pupDisk.def.h"
