//! main char object.  Creates array, handles command line, catches reduction.

#include <cmath>

class main : public Chare
{
private:
  CProxy_RedExample arr;
  double expected[2];
public:

  main(CkArgMsg* m);

  //! catches the completed reduction results.
  void reportIn(CkReductionMsg *msg)
  {
    int reducedArrSize=msg->getSize()/sizeof(double);
    CkAssert(reducedArrSize == 2);
    double *output=(double *)msg->getData();
    CkPrintf("Sum, Expectation, Difference:\n");
    for(int i=0;i<reducedArrSize;i++) {
      CkPrintf("%f %f %f\n", output[i], expected[i], output[i] - expected[i]);
      CkAssert(std::abs(output[i] - expected[i]) < 1e-7);
    }
    delete msg;
    done();
  }

  //! placeholder for a more interesting completion function
  void done()
  {
    CkPrintf("All done \n");
    CkExit();
  }

};

#include <vector>

//! RedExample. A small example which does nothing useful, but provides an extremely simple, yet complete, use of a normal reduction case.
/** The reduction performs a summation i+k1 i+k2 for i=0 to units
 */
class RedExample : public CBase_RedExample
{
 private:
  float myfloats[2];

 public:

  RedExample()   {   }

  RedExample(CkMigrateMessage *m) {};

  //! print our copy of the values.
  void dump()
    {
      CkPrintf("thisIndex %d %f %f\n",thisIndex,myfloats[0],myfloats[1]); 
    }


  //! add our index to the global floats and contribute the result.
  void dowork ()
    {
      std::vector<double> outdoubles(2);
      myfloats[0]=outdoubles[0]=   dOne +(double) thisIndex;
      myfloats[1]=outdoubles[1]= dTwo +(double) thisIndex;
      CkCallback cb(CkIndex_main::reportIn(NULL), mainProxy);
      contribute(outdoubles, CkReduction::sum_double, cb);
      dump();
    }
};

