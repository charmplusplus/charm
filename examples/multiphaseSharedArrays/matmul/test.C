// -*- mode: c++; tab-width: 4 -*-
#include "msa/msa.h"

#include <math.h>

const double epsilon = 0.00000001;

// ensure length is a multiple of 8, so that we can run this program on even number
// of PE's
// num data items in shared array
const unsigned int len = 200000;
// num BYTES in local cache
const unsigned int bytes = 4*1024*1024;

int notequal(double v1, double v2)
{
    double abs1 = fabs(v1 - v2);
    double abs2 = fabs(v2 - v1);

    // pick the worst case difference
    double absv = (abs1 < abs2) ? abs2 : abs1;

    return (absv > epsilon);
}


void mainFn()
{
    // <data type, page size> (num data items in shared array, num BYTES in local cache)
    MSA<double, 256> arr1(len, bytes);

    ckout <<  "--- Started processing in MainChare ---" << endl;
    //CthPrintThreadId(CthSelf());

    for(unsigned int i = 0; i < len; i++)
        arr1.set(i) = 44.6;

    arr1.sync(1);

    int msg = 1;
    for(unsigned int i = 0; i < len; i++)
        if(notequal(arr1.get(i), 44.6))
        {
            ckout << "MainChare: Inconsistent element " << i << ", value = " << arr1.get(i) << endl;
            msg = 0;
        }

    ckout << "--- MainChare: Done! ---" << endl;
}

#include "Test.decl.h"

class Test : public CBase_Test
{
protected:
    unsigned int doneCnt;
    double start_time, end_time;

public:
    Test(CkArgMsg* m) : doneCnt(0)
    {
        delete m;
        mainFn();

        // now create a distributed array
        MSA<double, 4096> arr2(len, bytes);
        MSA<double, 4096> arr3(2*len, bytes/2);
        MSA<double, 4096> arr4(4*len, 2*bytes);

        start_time = CkWallTimer();
        //CProxy_TestArray arr = CProxy_TestArray::ckNew(/*arr2.getCacheGroup(),*/ numThreads);
        //arr.ckSetReductionClient(new CkCallback(CkIndex_Test::done(), thisProxy));
        //done();
        CProxy_TestGroup::ckNew(thisProxy, arr2.getCacheGroup(), arr3.getCacheGroup(), arr4.getCacheGroup());
    }

    void done()
    {
        CkExit();
    }

    void doneg()
    {
        doneCnt++;
        if(doneCnt == CkNumPes())
        {
            end_time = CkWallTimer();
            ckout << "Done! Time required = " << end_time - start_time << endl;
            CkExit();
        }
    }
};

const int ok_message = 0;

class TestGroup : public CBase_TestGroup
{
protected:
    inline void TestWriteOnce(MSA<double, 4096>& arr1, double val = 0.0)
    {
        unsigned int mySectionSize = arr1.length()/CkNumPes();
        for(unsigned int i = CkMyPe()*mySectionSize; i < (CkMyPe() + 1)*mySectionSize; i++)
            arr1.set(i) = i*val;

        arr1.sync();

        int msg = 1;
        unsigned int len = arr1.length();
        for(unsigned int i = 0; i < len; i++)
            if(notequal(arr1.get(i), i*val) && msg)
            {
                ckout << "[" << CkMyPe() << "]Inconsistent element " << i << ", value = " << arr1.get(i) << endl;
                msg = 0;
            }

        if(msg && ok_message) ckout << "[" << CkMyPe() << "]WriteOnce OK" << endl;
    }

    inline void TestAccumulate(MSA<double, 4096>& arr1, double contrib)
    {
        //ckout << "[" << CkMyPe() << "] started sync request" << endl;
        arr1.sync();
        //ckout << "[" << CkMyPe() << "] sync request done" << endl;

        // test the accumulate interface
        for(unsigned int i = 0; i < arr1.length(); i++)
        {
            //arr1.set(i) = 0;
            arr1.accumulate(i, contrib);
        }

        arr1.sync();

        int msg = 1;
        for(unsigned int i = 0; i < arr1.length(); i++)
            if(notequal(arr1.get(i), contrib*CkNumPes()) && msg)
            {
                ckout << "[" << CkMyPe() << "]Inconsistent element " << i << ", value = " << arr1.get(i)
                      << ", expected = " << contrib*CkNumPes() << endl;
                msg = 0;
            }

        if(msg && ok_message) ckout << "[" << CkMyPe() << "]Accumulate " << contrib << " OK" << endl;
    }

public:
    TestGroup(CProxy_Test mainChare, CProxy_CacheGroup cg1, CProxy_CacheGroup cg2, CProxy_CacheGroup cg3)
    {
        if(ok_message) ckout << "Starting processing in group" << endl;

        //CthPrintThreadId(CthSelf());
        MSA<double, 4096> arr1(cg1);
        MSA<double, 4096> arr2(cg2);
        MSA<double, 4096> arr3(cg3);


        for(int i = 0; i < 5; i++)
        {
            TestWriteOnce(arr1, (double)i*234);
            TestWriteOnce(arr2, (double)i + 66.23948);
            TestWriteOnce(arr3, (double)i / 74.8 + 55e33);
        }


        double contribs[] = { 6.72, 4.66, 9.3200, 8.33, 89.434, 11.33 };

        for(unsigned int i = 0 ; i < sizeof(contribs)/sizeof(*contribs); i++)
        {
            if(CkMyPe() == 0 && ok_message) ckout << "//////// Iteration " << i << " //////////////" << endl;
            TestAccumulate(arr1, contribs[i]);
            TestWriteOnce(arr2);
            TestAccumulate(arr3, contribs[i]);
            TestWriteOnce(arr2);
            TestAccumulate(arr1, contribs[i]);
            TestAccumulate(arr2, contribs[i]);
            TestWriteOnce(arr3, 6.0);
            TestWriteOnce(arr1, 12.0e22);
        }

        if(ok_message) ckout << "[" << CkMyPe() << "] Done!" << endl;
        mainChare.doneg();
    }
};

#include "Test.def.h"

