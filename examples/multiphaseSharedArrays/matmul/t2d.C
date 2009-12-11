// -*- mode: c++; tab-width: 4 -*-

// When running 1D, make NEPP = COL1
// When running 2D, same
// When running 3D, make NEPP = subset of COL1

#include "nepp.h"
#include "msa/msa.h"

#ifdef PUP_EVERY
typedef MSA2D<double, DefaultEntry<double,true>, NEPP, MSA_ROW_MAJOR> MSA2DRowMjr;
#ifdef OLD
typedef MSA2D<double, DefaultEntry<double,true>, NEPP, MSA_COL_MAJOR> MSA2DColMjr;
#else
typedef MSA2D<double, DefaultEntry<double,true>, NEPP, MSA_ROW_MAJOR> MSA2DColMjr;
#endif
typedef MSA2D<double, DefaultEntry<double,true>, NEPP_C, MSA_ROW_MAJOR> MSA2DRowMjrC;

#else
typedef MSA2D<double, DefaultEntry<double,false>, NEPP, MSA_ROW_MAJOR> MSA2DRowMjr;
#ifdef OLD
typedef MSA2D<double, DefaultEntry<double,false>, NEPP, MSA_COL_MAJOR> MSA2DColMjr;
#else
typedef MSA2D<double, DefaultEntry<double,false>, NEPP, MSA_ROW_MAJOR> MSA2DColMjr;
#endif
typedef MSA2D<double, DefaultEntry<double,false>, NEPP_C, MSA_ROW_MAJOR> MSA2DRowMjrC;
#endif

#include "t2d.decl.h"

#include <assert.h>
#include <math.h>
#include "params.h"

const double epsilon = 0.00000001;
inline int notequal(double v1, double v2)
{
    return (fabs(v1 - v2) > epsilon);
}

static void perfCheck(int expected, int actual)
{
  if (expected != actual)
	ckout << "Arguments don't line up with compiled in defaults." << endl
		  << expected << " != " << actual << endl
		  << " Performance may suffer" << endl;
}

class t2d : public CBase_t2d
{
protected:
    double start_time;
    CProxy_TestArray workers;
    int reallyDone;

public:
    t2d(CkArgMsg* m)
    {
        // Usage: a.out number_of_worker_threads max_bytes ROWS1 ROWS2 COLS2 DECOMP-D TIMING-DETAIL?
        if(m->argc >1 ) NUM_WORKERS=atoi(m->argv[1]);
        if(m->argc >2 ) bytes=atoi(m->argv[2]);
        if(m->argc >3 ) ROWS1=atoi(m->argv[3]);
        if(m->argc >4 ) ROWS2=COLS1=atoi(m->argv[4]);
        if(m->argc >5 ) COLS2=atoi(m->argv[5]);
        if(m->argc >6 ) DECOMPOSITION=atoi(m->argv[6]); // 1D, 2D, 3D
        if(m->argc >7 ) detailedTimings= ((atoi(m->argv[7])!=0)?true:false);
        delete m;
        reallyDone = 0;

        MSA2DRowMjr arr1(ROWS1, COLS1, NUM_WORKERS, bytes);        // row major
        MSA2DColMjr arr2(ROWS2, COLS2, NUM_WORKERS, bytes);        // column major
        MSA2DRowMjrC prod(ROWS1, COLS2, NUM_WORKERS, bytes);        // product matrix

        workers = CProxy_TestArray::ckNew(arr1, arr2, prod, NUM_WORKERS, NUM_WORKERS);
        workers.ckSetReductionClient(new CkCallback(CkIndex_t2d::done(NULL), thisProxy));

        start_time = CkWallTimer();
        workers.Start();
    }

    // This method gets called twice, and should only terminate the
    // second time.
    void done(CkReductionMsg* m)
    {
        int *ip = (int*)m->getData();
        bool prefetchWorked = (*ip==0);
        delete m;

        if (reallyDone == 0) {
            workers.Kontinue();
            reallyDone++;

            double end_time = CkWallTimer();

            const char TAB = '\t';

            char hostname[100];
            gethostname(hostname, 100);

            ckout << CkNumPes() << TAB
				  << ROWS1 << TAB
                  << COLS1 << TAB
                  << ROWS2 << TAB
                  << COLS2 << TAB
                  << NUM_WORKERS << TAB
                  << bytes << TAB
                  << (runPrefetchVersion? (prefetchWorked?"Y":"N"): "U") << TAB
                  << end_time - start_time << TAB
                  << NEPP << TAB
                  << DECOMPOSITION << TAB
                  << hostname
                  << endl;

        } else {
            CkExit();
        }
    }
};

// get the chunk for a given index
int GetChunkForIndex(int index, int maxIndex, int numWorkers)
{
    int rangeSize = maxIndex / numWorkers;
    int chunk;

    // find which chare is going to process the current node
    if(index <= (maxIndex % numWorkers) * (rangeSize + 1) - 1)
        chunk = index/(rangeSize + 1);
    else
        chunk = maxIndex%numWorkers + (index - (maxIndex%numWorkers) * (rangeSize + 1))/rangeSize;

    return chunk;
}

// Returns start and end
void GetMyIndices(unsigned int maxIndex, unsigned int myNum, unsigned int numWorkers,
                  unsigned int& start, unsigned int& end)
{
    int rangeSize = maxIndex / numWorkers;
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
}

// class MatmulHelper {
// public:
//     unsigned int iStart, iEnd, jStart, jEnd, kStart, kEnd;
//     MatmulHelper(unsigned int ROWS1_, unsigned int COLS1_, unsigned int COLS2_)
//         : iStart(0), iEnd(ROWS1_-1),  // A's rows
//           kStart(0), kEnd(COLS1_-1),  // inner
//           jStart(0), jEnd(COLS2_-1)   // B's cols
//     {}
// }

// class MatmulHelper1D {
    
// }

class TestArray : public CBase_TestArray
{
private:
    // prefetchWorked keeps track of whether the prefetches succeeded or not.
    bool prefetchWorked;
    CkVec<double> times;
    CkVec<const char*> description;

    // ================================================================
    // 2D calculations

    inline int numWorkers2D() {
        static int n = 0;

        if (n==0) {
            n = (int)(sqrt(numWorkers));
            CkAssert(n*n == numWorkers);
        }

        return n;
    }

    // Convert a 1D ChareArray index into a 2D x dimension index
    inline unsigned int toX() {
        return thisIndex/numWorkers2D();
    }
    // Convert a 1D ChareArray index into a 2D y dimension index
    inline unsigned int toY() {
        return thisIndex%numWorkers2D();
    }

    // ================================================================
    // 3D calculations
    inline int numWorkers3D() {
        static int n = 0;

        if (n==0) {
            n = (int)(cbrt(numWorkers));
            CkAssert(n*n*n == numWorkers);
        }

        return n;
    }

    // Convert a 1D ChareArray index into a 3D x dimension index
    inline unsigned int toX3D() {
        int b = (numWorkers3D()*numWorkers3D());
        return thisIndex/b;
    }
    // Convert a 1D ChareArray index into a 3D y dimension index
    inline unsigned int toY3D() {
        int b = (numWorkers3D()*numWorkers3D());
        return (thisIndex%b)/numWorkers3D();
    }
    // Convert a 1D ChareArray index into a 3D z dimension index
    inline unsigned int toZ3D() {
        int b = (numWorkers3D()*numWorkers3D());
        return (thisIndex%b)%numWorkers3D();
    }

    // ================================================================

protected:
    MSA2DRowMjr arr1;       // row major
	MSA2DRowMjr::Read *h1;
    MSA2DColMjr arr2;       // column major
	MSA2DColMjr::Read *h2;
    MSA2DRowMjrC prod;       // product matrix
	MSA2DRowMjrC::Handle *hp;

    unsigned int rows1, rows2, cols1, cols2, numWorkers;

    void EnrollArrays()
    {
        arr1.enroll(numWorkers); // barrier
        arr2.enroll(numWorkers); // barrier
        prod.enroll(numWorkers); // barrier
    }

    void FillArrays(MSA2DRowMjr::Write &w1, MSA2DColMjr::Write &w2)
    {
        // fill in our portion of the array
        unsigned int rowStart, rowEnd, colStart, colEnd;
        GetMyIndices(rows1, thisIndex, numWorkers, rowStart, rowEnd);
        GetMyIndices(cols2, thisIndex, numWorkers, colStart, colEnd);

        // fill them in with 1
        for(unsigned int r = rowStart; r <= rowEnd; r++)
            for(unsigned int c = 0; c < cols1; c++)
                w1.set(r, c) = 1.0;

        for(unsigned int c = colStart; c <= colEnd; c++)
            for(unsigned int r = 0; r < rows2; r++)
                w2.set(r, c) = 1.0;

    }

    void FillArrays2D(MSA2DRowMjr::Write &w1, MSA2DColMjr::Write &w2)
    {
        unsigned int rowStart, rowEnd, colStart, colEnd;
        unsigned int r, c;

        // fill in our portion of the A matrix
        GetMyIndices(rows1, toX(), numWorkers2D(), rowStart, rowEnd);
        GetMyIndices(cols1, toY(), numWorkers2D(), colStart, colEnd);
        // CkPrintf("p%dw%d: FillArray2D A = %d %d %d %d\n", CkMyPe(), thisIndex, rowStart, rowEnd, colStart, colEnd);

        // fill them in with 1
        for(r = rowStart; r <= rowEnd; r++)
            for(c = colStart; c <= colEnd; c++)
                w1.set(r, c) = 1.0;

        // fill in our portion of the B matrix
        GetMyIndices(rows2, toX(), numWorkers2D(), rowStart, rowEnd);
        GetMyIndices(cols2, toY(), numWorkers2D(), colStart, colEnd);
        // CkPrintf("p%dw%d: FillArray2D B = %d %d %d %d\n", CkMyPe(), thisIndex, rowStart, rowEnd, colStart, colEnd);
        // fill them in with 1
        for(r = rowStart; r <= rowEnd; r++)
            for(c = colStart; c <= colEnd; c++)
                w2.set(r, c) = 1.0;
    }

    void TestResults(MSA2DRowMjr::Read &r1, MSA2DColMjr::Read &r2, MSA2DRowMjrC::Read &rp,
					 bool prod_test=true)
    {
        int errors = 0;
        bool ok=true;

        // verify the results, print out first error only
        ok=true;
        for(unsigned int r = 0; ok && r < rows1; r++) {
            for(unsigned int c = 0; ok && c < cols1; c++) {
                if(notequal(r1.get(r, c), 1.0)) {
                    ckout << "[" << CkMyPe() << "," << thisIndex << "] arr1 -- Illegal element at (" << r << "," << c << ") " << r1.get(r,c) << endl;
                    ok=false;
                    errors++;
                }
            }
        }

        ok=true;
        for(unsigned int c = 0; ok && c < cols2; c++) {
            for(unsigned int r = 0; ok && r < rows2; r++) {
                if(notequal(r2.get(r, c), 1.0)) {
                    ckout << "[" << CkMyPe() << "," << thisIndex << "] arr2 -- Illegal element at (" << r << "," << c << ") " << r2.get(r,c) << endl;
                    ok=false;
                    errors++;
                }
            }
        }

        //arr1.FreeMem();
        //arr2.FreeMem();

        if(prod_test)
        {
            ok = true;
            for(unsigned int c = 0; ok && c < cols2; c++) {
                for(unsigned int r = 0; ok && r < rows1; r++) {
                    if(notequal(rp.get(r,c), 1.0 * cols1)) {
                        ckout << "[" << CkMyPe() << "] result  -- Illegal element at (" << r << "," << c << ") " << rp.get(r,c) << endl;
                        ok=false;
                        errors++;
                    }
                }
            }
        }

        if (errors!=0) CkAbort("Incorrect array elements detected!");
    }

    void Contribute()
    {
        int dummy = prefetchWorked?0:1;
        contribute(sizeof(int), &dummy, CkReduction::sum_int);
    }

    // ============================= 1D ===================================

    void FindProductNoPrefetch(MSA2DRowMjr::Read &r1,
							   MSA2DColMjr::Read &r2,
							   MSA2DRowMjrC::Write &wp)
	{
#ifdef OLD
        FindProductNoPrefetchNMK(r1, r2, wp);
#else
        FindProductNoPrefetchMKN_RM(r1, r2, wp);
#endif
    }

    // new, but bad perf
    // improved perf by taking the prod.accu out of the innermost loop, up 2
    // further improved perf by taking the arr1.get out of the innermost loop, up 1.
    void FindProductNoPrefetchMKN_RM(MSA2DRowMjr::Read &r1,
									 MSA2DColMjr::Read &r2,
									 MSA2DRowMjrC::Write &wp)
    {
        CkAssert(arr2.getArrayLayout() == MSA_ROW_MAJOR);
//         CkPrintf("reached\n");
        unsigned int rowStart, rowEnd;
        GetMyIndices(rows1, thisIndex, numWorkers, rowStart, rowEnd);

        double *result = new double[cols2];
        for(unsigned int r = rowStart; r <= rowEnd; r++) { // M
            for(unsigned int c = 0; c < cols2; c++)
                result[c] = 0;
            for(unsigned int k = 0; k < cols1; k++) { // K
                double a = r1.get(r,k);
                for(unsigned int c = 0; c < cols2; c++) { // N
                    result[c] += a * r2.get(k,c);
//                     prod.set(r,c) = result; // @@ to see if accu is the delay
//                     prod.accumulate(prod.getIndex(r,c), result);
                }
//              assert(!notequal(result, 1.0*cols1));
            }
            for(unsigned int c = 0; c < cols2; c++) {
                wp.set(r,c) = result[c];
            }
        }
        delete [] result;
    }

    // old
    void FindProductNoPrefetchNMK(MSA2DRowMjr::Read &r1,
								  MSA2DColMjr::Read &r2,
								  MSA2DRowMjrC::Write &wp)
    {
        unsigned int rowStart, rowEnd;
        GetMyIndices(rows1, thisIndex, numWorkers, rowStart, rowEnd);

        for(unsigned int c = 0; c < cols2; c++) { // N
            for(unsigned int r = rowStart; r <= rowEnd; r++) { // M

                double result = 0.0;
                for(unsigned int k = 0; k < cols1; k++) { // K
                    double e1 = r1.get(r,k);
                    double e2 = r2.get(k,c);
                    result += e1 * e2;
                }
//              assert(!notequal(result, 1.0*cols1));

                wp.set(r,c) = result;
            }
        }
    }

    // Assumes that the nepp equals the size of a row, i.e. NEPP == COLS1 == ROWS2
    void FindProductNoPrefetchStripMined(MSA2DRowMjrC::Write &wp)
    {
        FindProductNoPrefetchStripMinedMKN_ROWMJR(wp);
    }

    // Assumes that the nepp equals the size of a row, i.e. NEPP == COLS1 == ROWS2
    void FindProductNoPrefetchStripMinedNMK(MSA2DRowMjrC::Write &wp)
    {
	    perfCheck(NEPP, cols1);
        unsigned int rowStart, rowEnd;
        GetMyIndices(rows1, thisIndex, numWorkers, rowStart, rowEnd);
        CkPrintf("p%dw%d: FPNP2DSM A = %d %d %d %d\n", CkMyPe(), thisIndex, rowStart, rowEnd, 0, cols2-1);

        double time1 = CmiWallTimer();
        for(unsigned int c = 0; c < cols2; c++) { // N
            for(unsigned int r = rowStart; r <= rowEnd; r++) {  // M

                double* a = &(arr1.getPageBottom(arr1.getIndex(r,0),Read_Fault)); // ptr to row of A
                double* b = &(arr2.getPageBottom(arr2.getIndex(0,c),Read_Fault)); // ptr to col of B
                double result = 0.0;
                for(unsigned int k = 0; k < cols1; k++) { // K
                    double e1 = a[k];  // no get
                    double e2 = b[k];  // no get
                    result += e1 * e2;
                }
//              assert(!notequal(result, 1.0*cols1));

                wp.set(r,c) = result;
            }
        }

        double time2 = CmiWallTimer();
        CkPrintf("timings %f \n", time2-time1);
    }

    // Assumes that the nepp equals the size of a row, i.e. NEPP == COLS1 == ROWS2
    // Assumes CkAssert(NEPP_C == cols2);
    void FindProductNoPrefetchStripMinedMKN_ROWMJR(MSA2DRowMjrC::Write &wp)
    {
	    perfCheck(NEPP, cols1);
	    perfCheck(NEPP_C, cols2);
        CkAssert(arr2.getArrayLayout() == MSA_ROW_MAJOR);
        unsigned int rowStart, rowEnd;
        GetMyIndices(rows1, thisIndex, numWorkers, rowStart, rowEnd);
//         CkPrintf("p%dw%d: FPNP1DSM_MKN_RM A = %d %d\n", CkMyPe(), thisIndex, rowStart, rowEnd);

        for(unsigned int r = rowStart; r <= rowEnd; r++) {  // M
            double* a = &(arr1.getPageBottom(arr1.getIndex(r,0),Read_Fault)); // ptr to row of A
            for(unsigned int c = 0; c < cols2; c++) { // N
                wp.set(r,c);  // just mark it as updated, need a better way
            }
            double* cm = &(prod.getPageBottom(prod.getIndex(r,0),Write_Fault)); // ptr to row of C
            for(unsigned int k = 0; k < cols1; k++) { // K
                double* b = &(arr2.getPageBottom(arr2.getIndex(k,0),Read_Fault)); // ptr to row of B
                for(unsigned int c = 0; c < cols2; c++) { // N
                    cm[c] += a[k]*b[c];
//                     prod.accumulate(prod.getIndex(r,c), );
                }
            }
			if (r%4==0) CthYield();
        }
    }

    void FindProductWithPrefetch(MSA2DRowMjr::Read &r1,
								 MSA2DColMjr::Read &r2,
								 MSA2DRowMjrC::Write &wp)
    {
        // fill in our portion of the array
        unsigned int rowStart, rowEnd;
        GetMyIndices(rows1, thisIndex, numWorkers, rowStart, rowEnd);

        arr1.Unlock(); arr2.Unlock();
        prefetchWorked = false;

        arr1.Prefetch(rowStart, rowEnd);
        arr2.Prefetch(0, cols2);

        /* if prefetch fails, switch to non-prefetching version */
        if(arr1.WaitAll())
        {
            if(verbose) ckout << thisIndex << ": Out of buffer in prefetch 1" << endl;
            FindProductNoPrefetch(r1, r2, wp);
            return;
        }

        if(arr2.WaitAll())
        {
            if(verbose) ckout << thisIndex << ": Out of buffer in prefetch 2" << endl;
            FindProductNoPrefetch(r1, r2, wp);
            return;
        }

        prefetchWorked = true;

        for(unsigned int c = 0; c < cols2; c++)
        {
            for(unsigned int r = rowStart; r <= rowEnd; r++)
            {
                double result = 0.0;
                for(unsigned int k = 0; k < cols1; k++)
                {
                    double e1 = r1.get2(r,k);
                    double e2 = r2.get2(k,c);
                    result += e1 * e2;
                }

                //ckout << "[" << r << "," << c << "] = " << result << endl;

                wp(r,c) = result;
            }
            //ckout << thisIndex << "." << endl;
        }

        //arr1.Unlock(); arr2.Unlock();
    }

    // ============================= 2D ===================================
    void FindProductNoPrefetch2DStripMined(MSA2DRowMjrC::Write &wp)
    {
        perfCheck(NEPP, cols1);
        unsigned int rowStart, rowEnd, colStart, colEnd;
        // fill in our portion of the C matrix
        GetMyIndices(rows1, toX(), numWorkers2D(), rowStart, rowEnd);
        GetMyIndices(cols2, toY(), numWorkers2D(), colStart, colEnd);

        for(unsigned int c = colStart; c <= colEnd; c++) {
            for(unsigned int r = rowStart; r <= rowEnd; r++) {

                double* a = &(arr1.getPageBottom(arr1.getIndex(r,0),Read_Fault)); // ptr to row of A
                double* b = &(arr2.getPageBottom(arr2.getIndex(0,c),Read_Fault)); // ptr to col of B

                double result = 0.0;
                for(unsigned int k = 0; k < cols1; k++) {
                    double e1 = a[k];
                    double e2 = b[k];
                    result += e1 * e2;
                }
//              assert(!notequal(result, 1.0*cols1));

                wp.set(r,c) = result;
            }
        }
    }

    void FindProductNoPrefetch2D(MSA2DRowMjr::Read &r1,
								 MSA2DColMjr::Read &r2,
								 MSA2DRowMjrC::Write &wp)
    {
        unsigned int rowStart, rowEnd, colStart, colEnd;
        // fill in our portion of the C matrix
        GetMyIndices(rows1, toX(), numWorkers2D(), rowStart, rowEnd);
        GetMyIndices(cols2, toY(), numWorkers2D(), colStart, colEnd);

        for(unsigned int c = colStart; c <= colEnd; c++) {
            for(unsigned int r = rowStart; r <= rowEnd; r++) {

                double result = 0.0;
                for(unsigned int k = 0; k < cols1; k++) {
                    double e1 = r1.get(r,k);
                    double e2 = r2.get(k,c);
                    result += e1 * e2;
                }
//              assert(!notequal(result, 1.0*cols1));

                wp.set(r,c) = result;
            }
        }
    }

    // ============================= 3D ===================================
    void FindProductNoPrefetch3D(MSA2DRowMjr::Read &r1,
								 MSA2DColMjr::Read &r2,
								 MSA2DRowMjrC::Accum &ap)
    {
        unsigned int rowStart, rowEnd, colStart, colEnd, kStart, kEnd;
        // fill in our portion of the C matrix
        GetMyIndices(rows1, toX3D(), numWorkers3D(), rowStart, rowEnd);
        GetMyIndices(cols2, toY3D(), numWorkers3D(), colStart, colEnd);
        GetMyIndices(cols1, toZ3D(), numWorkers3D(), kStart, kEnd);

        for(unsigned int c = colStart; c <= colEnd; c++) {
            for(unsigned int r = rowStart; r <= rowEnd; r++) {

                double result = 0.0;
                for(unsigned int k = kStart; k <= kEnd; k++) {
                    double e1 = r1.get(r,k);
                    double e2 = r2.get(k,c);
                    result += e1 * e2;
                }
//              assert(!notequal(result, 1.0*cols1));

                ap(r,c) += result;
            }
        }
    }

    void FindProductNoPrefetch3DStripMined(MSA2DRowMjrC::Accum &ap)
    {
        perfCheck(NEPP, cols1);
		unsigned int rowStart, rowEnd, colStart, colEnd, kStart, kEnd;
        // fill in our portion of the C matrix
        GetMyIndices(rows1, toX3D(), numWorkers3D(), rowStart, rowEnd);
        GetMyIndices(cols2, toY3D(), numWorkers3D(), colStart, colEnd);
        GetMyIndices(cols1, toZ3D(), numWorkers3D(), kStart, kEnd);

        for(unsigned int c = colStart; c <= colEnd; c++) {
            for(unsigned int r = rowStart; r <= rowEnd; r++) {

                double* a = &(arr1.getPageBottom(arr1.getIndex(r,0),Read_Fault)); // ptr to row of A
                double* b = &(arr2.getPageBottom(arr2.getIndex(0,c),Read_Fault)); // ptr to col of B
                double result = 0.0;
                for(unsigned int k = kStart; k <= kEnd; k++) {
                    double e1 = a[k];  // no get
                    double e2 = b[k];  // no get
                    result += e1 * e2;
                }
//              assert(!notequal(result, 1.0*cols1));

                ap(r,c) += result;
            }
        }
    }

    // ================================================================

public:
    TestArray(const MSA2DRowMjr &arr1_, const MSA2DColMjr &arr2_, MSA2DRowMjrC &prod_,
              unsigned int numWorkers_)
        : arr1(arr1_), arr2(arr2_), prod(prod_), numWorkers(numWorkers_), prefetchWorked(false),
          rows1(arr1.getRows()), cols1(arr1.getCols()),
          rows2(arr2.getRows()), cols2(arr2.getCols())
    {
        // ckout << "w" << thisIndex << ":" << rows1 << " " << cols1 << " " << cols2 << endl;
        times.push_back(CkWallTimer()); // 1
        description.push_back("constr");
    }

    TestArray(CkMigrateMessage* m) {}

    ~TestArray()
    {
    }

    void Start()
    {
        times.push_back(CkWallTimer()); // 2
        description.push_back("   start");

        EnrollArrays();
        times.push_back(CkWallTimer()); // 3
        description.push_back("   enroll");

		MSA2DRowMjr::Write &w1 = arr1.getInitialWrite();
		MSA2DColMjr::Write &w2 = arr2.getInitialWrite();

        if(verbose) ckout << thisIndex << ": filling" << endl;
        switch(DECOMPOSITION){
        case 1:
        case 3:
        case 4:
        case 6:
            FillArrays(w1, w2);
            break;
        case 2:
        case 5:
            FillArrays2D(w1, w2);
            break;
        }
        times.push_back(CkWallTimer()); // 4
        description.push_back("  fill");

        if(verbose) ckout << thisIndex << ": syncing" << endl;
        times.push_back(CkWallTimer()); // 5
        description.push_back("    sync");

//         if (do_test) TestResults(0);

        if(verbose) ckout << thisIndex << ": product" << endl;

		MSA2DRowMjr::Read &r1 = arr1.syncToRead(w1);
		MSA2DColMjr::Read &r2 = arr2.syncToRead(w2);

		hp = &(prod.getInitialWrite());
		MSA2DRowMjrC::Write &wp = * (MSA2DRowMjrC::Write *) hp;
		MSA2DRowMjrC::Accum &ap = * (MSA2DRowMjrC::Accum *) hp;

        switch(DECOMPOSITION) {
        case 1:
            if (runPrefetchVersion)
                FindProductWithPrefetch(r1, r2, wp);
            else
                FindProductNoPrefetch(r1, r2, wp);
            break;
        case 2:
            FindProductNoPrefetch2D(r1, r2, wp);
            break;
        case 3:
            FindProductNoPrefetch3D(r1, r2, ap);
            break;
        case 4:
            FindProductNoPrefetchStripMined(wp);
            break;
        case 5:
            FindProductNoPrefetch2DStripMined(wp);
            break;
        case 6:
            FindProductNoPrefetch3DStripMined(ap);
            break;
        }
        times.push_back(CkWallTimer()); // 6
        description.push_back("    work");

		h1 = &r1;
		h2 = &r2;
		hp = &(prod.syncToRead(*hp));

        Contribute();
    }

    void Kontinue()
    {
//         if (do_test) TestResults(0);
        times.push_back(CkWallTimer()); // 6
        description.push_back("    redn");

        if(verbose) ckout << thisIndex << ": testing" << endl;
        if (do_test) TestResults(*h1, *h2, * (MSA2DRowMjrC::Read *) hp);
        times.push_back(CkWallTimer()); // 5
        description.push_back("    test");
        Contribute();

        if (detailedTimings) {
            if (thisIndex == 0) {
                for(int i=1; i<description.length(); i++)
                    ckout << description[i] << " ";
                ckout << endl;
            }
            ckout << "w" << thisIndex << ":";
            for(int i=1; i<times.length(); i++)
                ckout << times[i]-times[i-1] << " ";
            ckout << endl;
        }
    }
};

#include "t2d.def.h"
