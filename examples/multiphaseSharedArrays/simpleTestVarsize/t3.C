// -*- mode: c++; tab-width: 4 -*-
#include "msa/msa.h"
class Double;
typedef MSA::MSA2D<Double, DefaultEntry<Double,true>, MSA_DEFAULT_ENTRIES_PER_PAGE, MSA_ROW_MAJOR> MSA2DRM;

#include "t3.decl.h"

#include <assert.h>
#include <math.h>
#include <iostream>
#include "params.h"

#define NO_PREFETCH
int g_prefetch = -1;

// debugging
#define XVAL 49
#define YVAL 76
const int do_message = 0;

const double epsilon = 0.00000001;
class Double {
    int getNumElements() {
        int i=0;
        Double *iter = this;
        while(iter!=0) {
            i++;
            iter = iter->next;
        }
        return i;
    }

public:
    double data;
    Double *next;

    // required
    Double()
    {
        data = 0.0;
        next = 0;
    }

    // optional, but recommended for user's code.
    // copy constructor
    //
	// Differs from copy assignment because cc deals with
	// unallocated memory, but ca deals with a constructed object.
    Double(const Double &rhs)
    {
        ckout << "reached copy" << endl;
//         data = rhs.data;
        next = 0;

        // call assignment operator
        *this = rhs;
    }

    ~Double()
    {
//         cout << "reached destructor" << endl;
        delete next;
    }

    // required
    // assignment operator
    Double& operator= (const Double& rhs)
    {
//         ckout << "reached assign" << endl;
        if (this == &rhs) return *this;  // self-assignment

        if (next != 0) {
            delete next;
            next = 0;
        }

        Double *iter1 = this;
        const Double *iter2 = &rhs;
        while (iter2 != 0) {
            iter1->data = iter2->data;
            if (iter2->next != 0)
                iter1->next = new Double();
            iter2 = iter2->next;
            iter1 = iter1->next;
        }

        return *this;
    }

    // required for accumulate
    // += operator
    // @@ what if rhs is a sequence.  Do we want to prepend the entire sequence to this Double?
    Double& operator+= (const Double& rhs)
    {
        if (rhs.data == 0) // identity
            return *this;
        else if (this->data == 0) {
            *this = rhs;
            return *this;
        }

        Double *last = this;
        Double *iter = this->next;
        while(iter!=0) {
            last = iter;
            iter = iter->next;
        }

        Double *tmp = new Double();
        last->next = tmp;
        *tmp = rhs; // use the assign operator to do the work.

        return *this;
    }

    // required for accumulate
    // typecast from int
    Double(const int rhs) : data(rhs), next(0)
    {
//         ckout << "reached typecast from int" << next << endl;
    }

    // required
    // pup
    virtual void pup(PUP::er &p){
//         static int called = 0;
//         called++;
//         ckout << "p" << CkMyPe() << ":" << "reached pup " << called << endl;

        if(false) { //simple pup
            p | data;
        } else {
            int n;
            if (p.isPacking())
                n = getNumElements();
            p|n;

            Double *iter = this;
            if (p.isUnpacking()) {
                CkAssert(0 == next);
                while (n>0) {
                    p|(iter->data);
                    n--;
                    if (n>0)
                        iter->next = new Double();
                    iter = iter->next;
                }
            } else {
                while(iter!=0) {
                    p|(iter->data);
                    iter = iter->next;
                }
            }
        }
    }

    // optional
    // typecast Double from/to double, for convenience
    Double(const double &rhs) : data(rhs), next(0) {}
//     operator double() { return data; }
//     operator double const () { return (const double) data; }
};

// optional
// convenience function
std::ostream& operator << (std::ostream& os, const Double& s) {
    os << s.data;
    if (s.next!=0)
        os << *(s.next);
    return os;
}

// optional
// convenience function
CkOutStream& operator << (CkOutStream& os, const Double& s) {
    os << s.data;
    if (s.next!=0)
        os << " " << *(s.next);
    return os;
}

inline int notequal(double v1, double v2)
{
    return (fabs(v1 - v2) > epsilon);
}

inline int notequal(Double v1, Double v2)
{
    if (notequal(v1.data, v2.data))
        return 1;
    else if (v1.next!=0 && v2.next!=0)
        return notequal(*v1.next, *v2.next);
    else 
        return !(v1.next == v2.next);
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
        } else {
            double end_time = CkWallTimer();

            const char TAB = '\t';

        ckout << ROWS1 << TAB
              << COLS1 << TAB
              << NUM_WORKERS << TAB
              << bytes << TAB
              << ((g_prefetch == 0) ? "N" : ((g_prefetch == 1) ? "Y" : "U")) << TAB
              << end_time - start_time
              << endl;

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

void GetMyIndices(unsigned int maxIndex, unsigned int myNum, unsigned int numWorkers, unsigned int& start, unsigned int& end)
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

class TestArray : public CBase_TestArray
{
protected:
    MSA2DRM arr1;       // row major
	MSA2DRM::Read r2;

    unsigned int rows1, cols1, numWorkers;

    void FillArray(MSA2DRM::Write &w)
    {
        // fill in our portion of the array
        unsigned int rowStart, rowEnd;
        GetMyIndices(rows1, thisIndex, numWorkers, rowStart, rowEnd);

        // fill them in with 1
        for(unsigned int r = rowStart; r <= rowEnd; r++)
            for(unsigned int c = 0; c < cols1; c++)
                w.set(r, c) = 1.0;

    }

    void FindProduct(MSA2DRM::Accum &a)
    {
        a(arr1.getIndex(0,0)) += 2.0 + thisIndex;
        a(arr1.getIndex(0,0)) += 100.0 + thisIndex;
    }

    void TestResults()
    {
		MSA2DRM::Read rh = r2;
        int error1 = 0, error2 = 0, error3=0;

        // verify the results
        int msg = 1;
        int cnt = 0;
        for(unsigned int r = 0; r < rows1; r++)
        {
            for(unsigned int c = 0; c < cols1; c++)
            {
                if(msg && notequal(rh.get(r, c).data, 1.0))
                {
                    ckout << "p" << CkMyPe() << "w" << thisIndex << " arr1 -- Illegal element at (" << r << "," << c << ") " << rh.get(r,c) << endl;
                    ckout << "Skipping rest of TestResults." << endl;
                    msg = 0;
                    error1 = 1;
                }
            }
        }

        if(do_message) ckout << "w" << thisIndex << ": Testing done.  Result = "
                             << ((error1 || error2 || error3)?"Failure":"SUCCESS")
                             << endl;
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
        if(do_message) ckout << "w" << thisIndex << ": filling" << endl;
		MSA2DRM::Write w = arr1.getInitialWrite();
        FillArray(w);
		r2 = w.syncToRead();
        if(do_message)
			ckout << "w" << thisIndex << ":value "
				  << r2.get(XVAL,YVAL) << "," << r2.get(XVAL,YVAL+1)  << endl;
//         (arr1.getCacheGroup()).emitBufferValue(6, 0);
        if(do_message)
			ckout << "w" << thisIndex << ": syncing" << endl;
//         if (thisIndex == 0) (arr1.getCacheGroup()).emit(0);
//         if(do_message) ckout << "w" << thisIndex << ":value2 " << arr1.get(XVAL,YVAL) << "," << arr1.get(XVAL,YVAL+1)  << endl;
        Contribute();
    }

    void Kontinue()
    {
//         if(do_message) ckout << "w" << thisIndex << ":value3 " << arr1.get(XVAL,YVAL) << "," << arr1.get(XVAL,YVAL+1)  << endl;
        if(do_message) ckout << thisIndex << ": testing after fillarray, sync, and redn" << endl;
        TestResults();

		MSA2DRM::Accum a = r2.syncToAccum();

        if(do_message) ckout << thisIndex << ": producting" << endl;
        FindProduct(a);

		r2 = a.syncToRead();
//         if(do_message) ckout << thisIndex << ": tetsing after product" << endl;
//      TestResults();

        // Print out the accumulated element.
        ckout << "p" << CkMyPe() << "w" << thisIndex << ":" << r2.get(0,0) << endl;

        Contribute();
    }
};

#include "t3.def.h"
