#include "TypedReduction.h"
#include <stdlib.h>

/*readonly*/ CProxy_Driver driverProxy;

Driver::Driver(CkArgMsg* args) {
    driverProxy = this;
    int array_size = 10;
    if (args->argc > 1) array_size = strtol(args->argv[1], NULL, 10);
    w = CProxy_Worker::ckNew(array_size);
    CkCallback *cb = new CkCallback(CkIndex_Driver::untyped_done(NULL), thisProxy);
    w.ckSetReductionClient(cb);
    w.reduce();
    delete args;
}

void Driver::untyped_done(CkReductionMsg* m) {
    int* output = (int*)m->getData();
    CkPrintf("Untyped Sum: %d\n", output[0]);
    delete m;
    CkCallback *cb = new CkCallback(
            CkReductionTarget(Driver, typed_done), thisProxy);
    w.ckSetReductionClient(cb);
    w.reduce();
}

void Driver::typed_done(int result)
{
    CkPrintf("Typed Sum: %d\n", result);
    CkCallback *cb = new CkCallback(
            CkReductionTarget(Driver, typed_array_done), thisProxy);
    w.ckSetReductionClient(cb);
    w.reduce_array();
}

void Driver::typed_array_done(int* results, int n)
{
    CkPrintf("Typed Sum: [ ");
    for (int i=0; i<n; ++i) CkPrintf("%d ", results[i]);
    CkPrintf("]\n");
    CkCallback *cb = new CkCallback(
            CkReductionTarget(Driver, typed_array_done2), thisProxy);
    w.ckSetReductionClient(cb);
    w.reduce_array();
}

void Driver::typed_array_done2(int x, int y, int z)
{
    CkPrintf("Typed Sum: (x, y, z) = (%d, %d, %d)\n", x, y, z);
    CkCallback *cb = new CkCallback(
            CkReductionTarget(Driver, typed_array_done3), thisProxy);
    w.ckSetReductionClient(cb);
    w.reduce_array_doubles();
}

void Driver::typed_array_done3(int n, double* results)
{
    CkPrintf("Typed Sum: [ ");
    for (int i=0; i<n; ++i) CkPrintf("%.5g ", results[i]);
    CkPrintf("]\n");

    CkPrintf("Random data for tuple & statistics reduction: ");
    w.reduce_tuple();
}

void Driver::tuple_reducer_done(CkReductionMsg* msg)
{
    CkPrintf("\n");
    CkReduction::tupleElement* results = nullptr;
    int num_reductions = 0;
    msg->toTuple(&results, &num_reductions);

    int min_result = *(int*)results[0].data;
    int max_result = *(int*)results[1].data;
    int sum_result = *(int*)results[2].data;
    CkReduction::statisticsElement& stats_result = *(CkReduction::statisticsElement*)results[3].data;
    CkPrintf("Tuple Reduction: Min: %d, Max: %d, Sum: %d\n",
             min_result, max_result, sum_result);
    CkPrintf("Statistics Reduction: Count: %d, Mean: %.2f, Standard Deviation: %.2f\n",
             stats_result.count, stats_result.mean, stats_result.stddev());

    delete[] results;
    CkExit();
}


Worker::Worker() { }

void Worker::reduce() {
    int contribution=1;
    contribute(1*sizeof(int), &contribution, CkReduction::sum_int); 
}

void Worker::reduce_array() {
    int contribution[3]={1,2,3};
    contribute(3*sizeof(int), contribution, CkReduction::sum_int); 
}

void Worker::reduce_array_doubles() {
    double contribution[3] = { 0.16180, 0.27182, 0.31415 };
    contribute(3*sizeof(double), contribution, CkReduction::sum_double);
}

void Worker::reduce_tuple() {
    int val = drand48() * 100.0 + drand48() * 100.0;
    CkPrintf("%d, ", val);
    CkReduction::statisticsElement stats(val);

    int tuple_size = 4;
    CkReduction::tupleElement tuple_reduction[] = {
        CkReduction::tupleElement(sizeof(int), &val, CkReduction::min_int),
        CkReduction::tupleElement(sizeof(int), &val, CkReduction::max_int),
        CkReduction::tupleElement(sizeof(int), &val, CkReduction::sum_int),
        CkReduction::tupleElement(sizeof(CkReduction::statisticsElement), &stats, CkReduction::statistics) };

    CkReductionMsg* msg = CkReductionMsg::buildFromTuple(tuple_reduction, tuple_size);

    CkCallback cb(CkReductionTarget(Driver, tuple_reducer_done), driverProxy);
    msg->setCallback(cb);
    contribute(msg);
}

#include "TypedReduction.def.h"
