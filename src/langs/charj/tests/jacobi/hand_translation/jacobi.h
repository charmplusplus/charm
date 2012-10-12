#ifndef __jacobi__
#define __jacobi__

/**************************************************************************
 * WARNING                                                                *
 **************************************************************************
 * This is a machine generated header file.                               *
 * It is not meant to be edited by hand and may be overwritten by charjc. *
 **************************************************************************/

//#define RAW_STENCIL
#define ITER 100
#define WORK  1

#include <charm++.h>
#include <string>
#include <vector>
#include <iostream>
#include <Array.h>

using std::string;
using std::vector;
using std::cout;
using std::endl;
using CharjArray::Array;
using CharjArray::Domain;
using CharjArray::Range;
using CharjArray::Matrix;
using CharjArray::Vector;

#include "jacobi.decl.h"

//#include "Main.decl.h"
class Main: public CBase_Main {
    public: Main(CkArgMsg* m);
            void finished();
    public: int num_finished;
};

/* Readonly variables */
extern CProxy_Main main;
extern CProxy_Chunk chunks;
extern double start_time;
//#include "jacobi_readonly.decl.h"

class Chunk: public CBase_Chunk {
    Chunk_SDAG_CODE
    private: Array<double, 2>* A;
    private: Array<double, 2>* B;
    private: double myMax;
    private: int myxdim, myydim, total, counter;
    public: Chunk(int t, int x, int y);
    private: void sendStrips();
    private: void doStencil();
    private: void doStencil_raw();
    private: void resetBoundary();
    public: void processStripFromLeft(Array<double> __s);
    public: void processStripFromRight(Array<double> __s);
    public: void pup(PUP::er& p);
    public: Chunk();
    protected: void constructorHelper();
    public: Chunk(CkMigrateMessage*);
    static bool _trace_registered;
    void _initTrace();
    int _sdag_jacobi_i;

    Array<double>* strip;
};
#endif // __jacobi__

