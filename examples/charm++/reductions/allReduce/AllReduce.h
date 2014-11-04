//! main char object.  Creates array, handles command line, catches reduction.
#include <ctime>

class main : public CBase_main
{
private:
  CProxy_AllReduce arr;
	double startTime;
public:

  main(CkArgMsg* m);

  void done()
  {
	  static int checkIn=0;
	  checkIn++;
	  if (checkIn==units) {
		  CkPrintf("All done in %lf seconds\n",(CkWallTimer()-startTime));
		  CkExit();  
	  }
  }

};


//! AllReduce. A small example which does nothing useful, but provides an extremely simple, yet complete, use of a normal reduction case.
/** The reduction performs a summation i+k1 i+k2 for i=0 to units
 */
class AllReduce : public CBase_AllReduce
{
 private:
	CProxy_main mainProxy;
	double* myData;
 public:

	AllReduce(CProxy_main ma)   { mainProxy=ma;  }

  AllReduce(CkMigrateMessage *m) {};

	void init()
	{
		                myData = new double[allredSize/(sizeof(double))];
				                for (int i=0; i<allredSize/sizeof(double); i++) {
							                        myData[i] = (double)i;
										                }
	}

  //! add our index to the global floats and contribute the result.
  void dowork ()
    {
	CkCallback cb(CkIndex_AllReduce::report(NULL),  thisProxy);
      contribute(allredSize,myData,CkReduction::sum_double,cb); 
    }
  //! catches the completed reduction results.
  void report(CkReductionMsg *msg)
  {
	  int reducedArrSize=msg->getSize()/sizeof(double);
	  double *output=(double *)msg->getData();
	  CkPrintf("done %d\n", thisIndex);
	  for(int i=0;i<reducedArrSize;i++) {
		  if (output[i]!=i*units) {
			  CkPrintf("result error!\n");
		  }
	  }
	  delete msg;
	  mainProxy.done();
  }

};

