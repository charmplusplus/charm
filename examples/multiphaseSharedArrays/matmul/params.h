unsigned int NUM_WORKERS = 2;
unsigned int bytes = 1024*1024;
unsigned int ROWS1 = 100;
unsigned int COLS1 = 500;
unsigned int COLS2 = 100;
unsigned int ROWS2 = COLS1;
unsigned int DECOMPOSITION = 1; // 1D matmul is the default, i.e. i=subset of ROWS1
// 4 = 1D stripmined
bool detailedTimings = false;

// Run the version without prefetching
const bool runPrefetchVersion=false;

// debugging
const bool verbose = false;
const bool do_test = true;  // If true, tests results, etc.

