//! main char object.  Creates array, handles command line, catches reduction.

class main : public Chare
{
private:
  CProxy_RedExample arr;
public:

  main(CkArgMsg* m);

  //! catches the completed reduction results.
  void reportIn(CkReductionMsg *msg)
  {
    int reducedArrSize=msg->getSize()/sizeof(double);
    double *output=(double *)msg->getData();
    CkPrintf("Sum :");
    for(int i=0;i<reducedArrSize;i++)
      CkPrintf("%f ",output[i]);
    CkPrintf("\n");
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
      double outdoubles[2];
      myfloats[0]=outdoubles[0]=   dOne +(double) thisIndex;
      myfloats[1]=outdoubles[1]= dTwo +(double) thisIndex;
      contribute(2*sizeof(double),outdoubles,CkReduction::sum_double); 
      dump();
    }
};

