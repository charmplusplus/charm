// -*- mode: c++; tab-width: 4 -*-
#include "msa/msa.h"


typedef MSA::MSA2D<double, DefaultEntry<double>, MSA_DEFAULT_ENTRIES_PER_PAGE, MSA_ROW_MAJOR> MSA2DRM;

#include "t3.decl.h"

#include <assert.h>
#include <math.h>
#include "params.h"

#define NO_PREFETCH
int g_prefetch = -1;

// debugging
const int do_message = 0;

const double epsilon = 0.00000001;
inline int notequal(double v1, double v2)
{
    return (fabs(v1 - v2) > epsilon);
}

class t3 : public CBase_t3
{
protected:
    double start_time;
    CProxy_TestArray workers;
    int reallyDone;

public:
    t3(CkArgMsg* m)
    {
        // Usage: t3 [number_of_worker_threads [max_bytes]]
        if(m->argc >1 ) NUM_WORKERS=atoi(m->argv[1]);
        if(m->argc >2 ) bytes=atoi(m->argv[1]);
        delete m;
        reallyDone = 0;

        // Actually build the shared array.
        MSA2DRM arr1(ROWS1, COLS1, NUM_WORKERS, bytes);

        workers = CProxy_TestArray::ckNew(arr1, NUM_WORKERS, NUM_WORKERS);
        workers.ckSetReductionClient(new CkCallback(CkIndex_t3::done(NULL), thisProxy));

        start_time = CkWallTimer();
        workers.Start();
    }

    void done(CkReductionMsg* m)
    {
        delete m;

        if (reallyDone == 0) {
            workers.Kontinue();
            reallyDone++;
            double end_time = CkWallTimer();

            const char TAB = '\t';

        ckout << ROWS1 << TAB
              << COLS1 << TAB
              << NUM_WORKERS << TAB
              << bytes << TAB
              << ((g_prefetch == 0) ? "N" : ((g_prefetch == 1) ? "Y" : "U")) << TAB
              << end_time - start_time
              << endl;
        } else {

            CkExit();
        }
    }
};


void GetMyIndices(unsigned int maxIndex, unsigned int myNum, unsigned int numWorkers, unsigned int& start, unsigned int& end)
{
    int rangeSize = maxIndex / numWorkers;
    start=myNum*rangeSize;
    end=(myNum+1)*rangeSize;
    if (myNum==numWorkers-1) end=maxIndex;

/*  // I don't understand what this is trying to do:
    if(myNum < maxIndex % numWorkers)
    {
        start = myNum * (rangeSize + 1);
        end = start + rangeSize;
    }
    else
    {
        start = myNum * rangeSize + maxIndex % numWorkers;
        end = start + rangeSize - 1;
    }
*/
}

class TestArray : public CBase_TestArray
{
protected:
    MSA2DRM arr1;       // row major

    unsigned int rows1, cols1, numWorkers;

    void FillArray(MSA2DRM::Write &w)
    {
        // fill in our portion of the array
        unsigned int rowStart, rowEnd;
        GetMyIndices(rows1, thisIndex, numWorkers, rowStart, rowEnd);

        // fill them in with 1
        for(unsigned int r = rowStart; r < rowEnd; r++)
            for(unsigned int c = 0; c < cols1; c++)
                w.set(r, c) = 1.0;
    }

    void TestResults(MSA2DRM::Read &rh)
    {
        int errors = 0;

        // verify the results
        for(unsigned int r = 0; r < rows1; r++)
        {
            bool warnedRow=false;
            for(unsigned int c = 0; c < cols1; c++)
            {
                if((!warnedRow) && notequal(rh.get(r, c), 1.0))
                {
                    ckout << "p" << CkMyPe() << "w" << thisIndex << " arr1 -- Illegal element at (" << r << "," << c << ") " << rh.get(r,c) << endl;
                    errors++;
                    warnedRow=true;
                }
            }
        }

        if (errors) CkAbort("Incorrect array elements detected!");
    }

    void Contribute()
    {
        int dummy = 0;
        contribute(sizeof(int), &dummy, CkReduction::max_int);
    }

public:
    TestArray(const MSA2DRM &arr_, unsigned int numWorkers_)
    : arr1(arr_), rows1(arr1.getRows()), cols1(arr1.getCols()), numWorkers(numWorkers_)
    {
    }

    TestArray(CkMigrateMessage* m) {}

    ~TestArray()
    {
    }

    // threaded EP
    void Start()
    {
        arr1.enroll(numWorkers); // barrier
		MSA2DRM::Write w = arr1.getInitialWrite();
        FillArray(w);
		MSA2DRM::Read r = w.syncToRead();
        TestResults(r);  // test before doing a reduction
        Contribute();
    }

    void Kontinue()
    {
//        TestResults();
        Contribute();
    }
};

#include "t3.def.h"
