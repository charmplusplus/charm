extern double WasteUnits;
extern CProxy_main mainProxy;
extern CProxy_ReadArrFour fourProxy;		
extern CProxy_ReadArrSix sixProxy;		
extern double dOne;
extern double dTwo;
extern CkVec<int> IntArrFour;
extern CkVec<int> IntArrFive;
extern CkVec<CProxy_groupTest> groupProxy;
extern CkVec<CProxy_groupTestX> groupProxyX;

bool CheckAllReadOnly();
void WasteTime(double);
class intdual {
 private:
    int x, y;
 public:
    intdual(){x=y=0;}

    intdual(int _x,int _y) : x(_x), y(_y){}
    void pup(PUP::er &p)
      {
	  p|x;
	  p|y;
      }
    inline int getx(){return x;};
    inline int gety(){return y;};
    inline CkHashCode hash() const {
	return (CkHashCode)((x<<10)+y);
    }
    static CkHashCode staticHash(const void *k,size_t){
	return ((intdual *)k)->hash();
    }
    inline int compare(intdual &t) const{
	return (t.getx() == x && t.gety() == y);
    }
    static int staticCompare(const void *a,const void *b,size_t){
	return ((intdual *)a)->compare((*(intdual *)b));
    }
   
};


class main : public CBase_main
{
 public:
  int doneCount;
  main(CkArgMsg *msg);
  void createReport(CkReductionMsg *msg);
  void doneReport(CkReductionMsg *msg);
};

class groupTest : public CBase_groupTest
{
 public:
  int order;
  groupTest(int order): order(order) 
    {
      // test that all lower order ids were created first
      for(int i=order-1;i>0;i--)
	{
	  //make a proxy
	  //get a branch
	  //test the order value in it
	  //	  CkPrintf("[%d] test group id %d looking at %d\n",CkMyPe(),order,i);
	  CkAssert(groupProxy[i].ckLocalBranch()->order==i);

	}
    }
};

class groupTestX : public CBase_groupTestX
{
 public:
  int order;
  groupTestX(int order): order(order) 
    {
      // test that all lower order ids were created first
      for(int i=order-1;i>0;i--)
	{
	  //make a proxy
	  //get a branch
	  //test the order value in it
	  //	  CkPrintf("[%d] test group id %d looking at %d\n",CkMyPe(),order,i);
	  CkAssert(groupProxy[i].ckLocalBranch()->order==i);
	  CkAssert(groupProxyX[i].ckLocalBranch()->order==i);

	}
    }
};

class ReadArrZero : public CBase_ReadArrZero
{
 private:
 public:
  int size;
  double units;
  bool workdone;
  bool reddone;
  ReadArrZero(int arrSize, double WasteUnits) :size(arrSize), units(WasteUnits)
    {
      int a=1;
      workdone=false;
      reddone=false;
      contribute(sizeof(int),&a,CkReduction::sum_int,CkCallback(CkIndex_main::createReport(NULL),mainProxy),0);
    }

  ReadArrZero()
    {
      workdone=false;
      reddone=false;
    }

  ReadArrZero(CkMigrateMessage *m) {};

  void receiveRed(CkReductionMsg *msg)
    {

      int *data   = (int *)msg->getData();
      int sender      = msg->getUserFlag();  

      CkPrintf("{%d} on [%d] gets reduction from %d, recv %d\n",thisIndex, CkMyPe(), sender, data[0]);
      CkAssert(sender==thisIndex+1);
      reddone=true;
      if(workdone)
	{
	  contribute(sizeof(int),data,CkReduction::sum_int,CkCallback(CkIndex_main::doneReport(NULL),mainProxy),0);
	}
      delete msg;
    }
  
  void dowork()
  {
    WasteTime(WasteUnits);
    int a=1;
    workdone=true;
    if(reddone || thisIndex>6)
      {
	contribute(sizeof(int),&a,CkReduction::sum_int,CkCallback(CkIndex_main::doneReport(NULL),mainProxy),0);
      }

  }

};

class ReadArrOne : public CBase_ReadArrOne
{
 private:
 public:
  int size;
  double units;
  CkCallback callbacktoZero;  
  
  ReadArrOne(int arrSize, double WasteUnits, CkCallback _cb) :size(arrSize), units(WasteUnits), callbacktoZero(_cb)
    {
      int a=1;
      contribute(sizeof(int),&a,CkReduction::sum_int,CkCallback(CkIndex_main::createReport(NULL),mainProxy),1);
    }

  ReadArrOne(CkMigrateMessage *m) {};
  
  void dowork(){
    WasteTime(WasteUnits);
    int a=1;
    contribute(sizeof(int),&a,CkReduction::sum_int,CkCallback(CkIndex_main::doneReport(NULL),mainProxy),1);
    contribute(sizeof(int), &a, CkReduction::sum_int,  callbacktoZero,1);

  }

};

class ReadArrTwo : public CBase_ReadArrTwo
{
 private:
 public:
  int size;
  double units;
  CkCallback callbacktoZero;

  ReadArrTwo(int arrSize, double WasteUnits, CkCallback _cb) :size(arrSize), units(WasteUnits), callbacktoZero(_cb)
    {
      int a=1;
      contribute(sizeof(int),&a,CkReduction::sum_int,CkCallback(CkIndex_main::createReport(NULL),mainProxy),2);
    }

  ReadArrTwo(CkMigrateMessage *m) {};

  
  void dowork(){
    WasteTime(WasteUnits);
    int a=1;
    contribute(sizeof(int),&a,CkReduction::sum_int,CkCallback(CkIndex_main::doneReport(NULL),mainProxy),2);
    contribute(sizeof(int), &a, CkReduction::sum_int,  callbacktoZero,2);
  }


};

class ReadArrThree : public CBase_ReadArrThree
{
 private:
 public:
  int size;
  double units;
  CkCallback callbacktoZero;
  ReadArrThree(int arrSize, double WasteUnits, CkCallback _cb) :size(arrSize), units(WasteUnits), callbacktoZero(_cb)
    {
      int a=1;
      if(CkMyPe()%2)
	WasteTime(WasteUnits);
      contribute(sizeof(int),&a,CkReduction::sum_int,CkCallback(CkIndex_main::createReport(NULL),mainProxy),3);
    }

  ReadArrThree(CkMigrateMessage *m) {};
  
  void dowork(){
    WasteTime(WasteUnits);
    int a=1;
    contribute(sizeof(int),&a,CkReduction::sum_int,CkCallback(CkIndex_main::doneReport(NULL),mainProxy),3);
    contribute(sizeof(int), &a, CkReduction::sum_int,  callbacktoZero,3);
  }


  
};

class ReadArrFour : public CBase_ReadArrFour
{
 private:
 public:
  int size;
  double units;
  int *data;
  CkCallback callbacktoZero;
  
  ReadArrFour(int arrSize, double WasteUnits, CkCallback _cb) :size(arrSize), units(WasteUnits), callbacktoZero(_cb) 
    {
      int a=1;
      data=new int[arrSize];
      memcpy(data,&(IntArrFour[0]),sizeof(int)*arrSize);
      contribute(sizeof(int),&a,CkReduction::sum_int,CkCallback(CkIndex_main::createReport(NULL),mainProxy),4);
    }

  ReadArrFour(CkMigrateMessage *m) {};

  
  void dowork()
  {
    WasteTime(WasteUnits);
    int a=1;
    contribute(sizeof(int),&a,CkReduction::sum_int,CkCallback(CkIndex_main::doneReport(NULL),mainProxy),4);
    contribute(sizeof(int), &a, CkReduction::sum_int,  callbacktoZero,4);
  }


  
};

class ReadArrFive : public CBase_ReadArrFive
{
 private:
  int *data;
 public:
  int size;
  double units;
  bool validate; 
  CkCallback callbacktoZero;
  ReadArrFive(int arrSize, double WasteUnits, bool validate, CkCallback _cb) :size(arrSize), units(WasteUnits), validate(validate), callbacktoZero(_cb) 
    {
      int a=1;
      // we shadow four, use its data member
      if(validate)
	{
	  data=fourProxy(thisIndex).ckLocal()->data;
	  CkAssert(data!=NULL);
	  for(int i=0;i<arrSize;i++)
	    CkAssert(data[i]==i);
	}
      contribute(sizeof(int),&a,CkReduction::sum_int,CkCallback(CkIndex_main::createReport(NULL),mainProxy),5);
    }
  
  ReadArrFive(CkMigrateMessage *m) {};

  void dowork(){
    WasteTime(WasteUnits);
    int a=1;
    contribute(sizeof(int),&a,CkReduction::sum_int,CkCallback(CkIndex_main::doneReport(NULL),mainProxy),5);
    contribute(sizeof(int), &a, CkReduction::sum_int,  callbacktoZero,5);

  }

};


class ReadArrSix : public CBase_ReadArrSix
{
 private:
 public:
  int *data;
  int size;
  int size2;
  double units;
  CkCallback callbacktoZero;
  ReadArrSix(int arrSize, int arrSize2, double WasteUnits, CkCallback _cb) :size(arrSize), size2(arrSize2), units(WasteUnits), callbacktoZero(_cb) 
    {
      int a=1;
      data=new int[arrSize];
      memcpy(data,&(IntArrFive[0]),sizeof(int)*arrSize);
      contribute(sizeof(int),&a,CkReduction::sum_int,CkCallback(CkIndex_main::createReport(NULL),mainProxy),6);
    }
  
  ReadArrSix(CkMigrateMessage *m) {};

  void dowork(){
    WasteTime(WasteUnits);
    int a=1;
    contribute(sizeof(int),&a,CkReduction::sum_int,CkCallback(CkIndex_main::doneReport(NULL),mainProxy),6);
    contribute(sizeof(int), &a, CkReduction::sum_int,  callbacktoZero,6);

  }

};

class ReadArrSeven : public CBase_ReadArrSeven
{
 private:
 public:
  int *data;
  int size;
  int size2;
  double units;
  bool validate;
  CkCallback callbacktoZero;

  ReadArrSeven(int arrSize, int arrSize2, double WasteUnits, bool validate, CkCallback _cb) :size(arrSize), size2(arrSize2), units(WasteUnits), validate(validate), callbacktoZero(_cb) 
    {
      int a=1;
      // we shadow six, use its data member
      if(validate)
	{
	  data=sixProxy(thisIndex.x,thisIndex.y).ckLocal()->data;
	  CkAssert(data!=NULL);
	  bool pass=true;
	  for(int i=0;i<arrSize;i++)
	    if(data[i]!=i)
	      {
		CkPrintf("[%d,%d] i %d != data[%d]\n",thisIndex.x, thisIndex.y,i,data[i]);
		pass=false;
	      }
	  CkAssert(pass);
	}

      contribute(sizeof(int),&a,CkReduction::sum_int,CkCallback(CkIndex_main::createReport(NULL),mainProxy),7);
    }
  
  ReadArrSeven(CkMigrateMessage *m) {};

  void dowork(){
    WasteTime(WasteUnits);
    int a=1;
    contribute(sizeof(int),&a,CkReduction::sum_int,CkCallback(CkIndex_main::doneReport(NULL),mainProxy),7);
    contribute(sizeof(int), &a, CkReduction::sum_int,  callbacktoZero,7);

  }

};



class OneMap: public CkArrayMap { 
 public:
  double howmuch;
  OneMap(double _howmuch): howmuch(_howmuch)
    {
      CheckAllReadOnly();
      WasteTime(howmuch);
    }
  int procNum(int, const CkArrayIndex &);
};

class TwoMap: public CkArrayMap { 
 public:
  double howmuch;
  TwoMap(double _howmuch): howmuch(_howmuch)
    {
      CheckAllReadOnly();
      WasteTime(howmuch);
    }
    int procNum(int, const CkArrayIndex &);
};

class ThreeMap: public CkArrayMap { 
 public:
  double howmuch;
  ThreeMap(double _howmuch) : howmuch(_howmuch)
    {
      CheckAllReadOnly();
      WasteTime(howmuch);
    }
    int procNum(int, const CkArrayIndex &);
};

class FourMap: public CkArrayMap { 
 public:
  double howmuch;
  FourMap(double _howmuch) : howmuch(_howmuch)
    {
      CheckAllReadOnly();
      WasteTime(howmuch);
    }
    int procNum(int, const CkArrayIndex &);
};

class FiveMap: public CkArrayMap { 
 public:
  double howmuch;
  FiveMap(double _howmuch) : howmuch(_howmuch)
    {
      CheckAllReadOnly();
      WasteTime(howmuch);
    }
    int procNum(int, const CkArrayIndex &);
};


class SixMap: public CkArrayMap { 
 public:
  double howmuch;
  SixMap(double _howmuch) : howmuch(_howmuch)
    {
      CheckAllReadOnly();
      WasteTime(howmuch);
    }
  int procNum(int, const CkArrayIndex &);
};
