// -*- mode: c++; tab-width: 4 -*-
//
#include "msa/msa.h"

typedef MSA::MSA2D<int, DefaultEntry<int>,
        MSA_DEFAULT_ENTRIES_PER_PAGE, MSA_ROW_MAJOR> MSA2D;
typedef MSA::MSA1D<int, DefaultEntry<int>, MSA_DEFAULT_ENTRIES_PER_PAGE> MSA1D;

#include "histogram.decl.h"


const unsigned int ROWS = 2000;
const unsigned int COLS = 2000;
const unsigned int BINS = 10;
const unsigned int MAX_ENTRY = 1000;
unsigned int WORKERS = 10;


class Driver : public CBase_Driver
{
  CProxy_Histogram workers;
public:
    Driver(CkArgMsg* m)
    {
        // Usage: histogram [number_of_worker_threads]
        if (m->argc > 1) WORKERS=atoi(m->argv[1]);
        delete m;

        // Actually build the shared arrays: a 2d array to hold arbitrary
        // data, and a 1d histogram array.
        MSA2D data(ROWS, COLS, WORKERS);
        MSA1D bins(BINS, WORKERS);
 
        // Create worker threads and start them off.
        workers = CProxy_Histogram::ckNew(data, bins, WORKERS);
        workers.ckSetReductionClient(
            new CkCallback(CkIndex_Driver::done(NULL), thisProxy));
        workers.start();
    }

    void done(CkReductionMsg* m)
    {
        // When the reduction is complete, everything is ready to exit.
        CkExit();
    }
};


class Histogram: public CBase_Histogram
{
public:
    MSA2D data;
    MSA1D bins;

    Histogram(const MSA2D& data_, const MSA1D& bins_)
    : data(data_), bins(bins_)
    {}

    Histogram(CkMigrateMessage* m)
    {}

    ~Histogram()
    {}

    // Note: it's important that start is a threaded entry method
    // so that the blocking MSA calls work as intended.
    void start()
    {
        data.enroll(WORKERS);
        bins.enroll(WORKERS);
        
        // Fill the data array with random numbers.
		MSA2D::Write wd = data.getInitialWrite();
        if (thisIndex == 0) fill_array(wd);

        // Fill the histogram bins: read from the data array and
        // accumulate to the histogram array.
		MSA2D::Read rd = wd.syncToRead();
        MSA1D::Accum ab = bins.getInitialAccum();
        fill_bins(ab, rd);

        // Print the histogram.
        MSA1D::Read rb = ab.syncToRead();
        if (thisIndex == 0) print_array(rb);

        // Contribute to Driver::done to terminate the program.
        contribute();
    }

    void fill_array(MSA2D::Write& w)
    {
        // Just let one thread fill the whole data array
        // with random entries to be histogrammed.
        // 
        // Note: this is potentially a very inefficient access
        // pattern, especially if the MSA doesn't fit into
        // memory, but it can be convenient.
        for (unsigned int r = 0; r < data.getRows(); r++) {
            for (unsigned int c = 0; c < data.getCols(); c++) {
                w.set(r, c) = random() % MAX_ENTRY;
            }
        }
    }

    void fill_bins(MSA1D::Accum& b, MSA2D::Read& d)
    {
        // Determine the range of the data array that this
        // worker should read from.
        unsigned int range = ROWS / WORKERS;
        unsigned int min_row = thisIndex * range;
        unsigned int max_row = (thisIndex + 1) * range;
        
        // Count the entries that belong to each bin and accumulate
        // counts into the bins.
        for (unsigned int r = min_row; r < max_row; r++) {
            for (unsigned int c = 0; c < data.getCols(); c++) {
                unsigned int bin = d.get(r, c) / (MAX_ENTRY / BINS);
                b(bin) += 1;
            }
        }
    }

    void print_array(MSA1D::Read& b)
    {
        for (unsigned int i=0; i<BINS; ++i) {
            CkPrintf("%d ", b.get(i)); 
        }
        CkPrintf("\n");
    }
};

#include "histogram.def.h"
