#include "jacobi.h"

/* Readonly variable support */
CProxy_Main main_proxy;
CProxy_Chunk chunks;
double start_time;

Main::Main(CkArgMsg* m)
{
    if (m->argc != 4) CkAbort("Wrong parameters\n");
    int x = atoi((m->argv)[1]);
    int y = atoi((m->argv)[2]);
    int k = atoi((m->argv)[3]);
    if (x < k) CkAbort("Xdim must be greater than k");
    if (k < CkNumPes() || k % CkNumPes()) CkAbort("k must be a multiple of numPes.");
    chunks = CProxy_Chunk::ckNew(k, x, y, k);
    chunks.jacobi();
    num_finished = k;
    main_proxy = thisProxy;
    start_time = CmiWallTimer();
}

void Main::finished()
{
    if (--num_finished == 0) {
        double elapsed = CmiWallTimer() - start_time;
        CkPrintf("Finished in %fs %fs/step, %d iterations\n", elapsed, elapsed/ITER, ITER);
        CkExit();
    }
}
//#include "Main.def.h"

//#include "jacobi_readonly.def.h"

//#include "Chunk.decl.h"
#if _CHARJ_TRACE_ALL_METHODS || _CHARJ_TRACE_TRACED_METHODS
#include <trace-projections.h>
#endif
Chunk::Chunk(int t, int x, int y)
{
    constructorHelper();
    int xdim, ydim;
    xdim = x;
    ydim = y;
    total = t;
    myMax = 99999.999;
    myxdim = (int)(xdim / total);
    if (thisIndex == total - 1) {
        myxdim = xdim - myxdim * (total - 1);
    }
    myydim = ydim;
    counter = 0;
    if (thisIndex != 0 && thisIndex != total - 1) {
        xdim = myxdim + 2;
    } else {
        xdim = myxdim + 1;
    }
    A = new Array<double, 2>(Domain<2>(Range(xdim), Range(myydim)));
    B = new Array<double, 2>(Domain<2>(Range(xdim), Range(myydim)));
    A->fill(0);
    B->fill(0);
    strip = new Array<double>(Domain<1>(Range(myydim)));
}

void Chunk::sendStrips()
{
        //Array<double>* dummy = new Array<double>(Domain<1>(Range(myydim)));
        //int left = thisIndex - 1;
        //int right = thisIndex + 1;
        //int max = total - 1;
        //if (thisIndex > 0) {
        //    chunks[left].getStripFromRight((A->access(1, Range(0,myydim))));
        //} else {
        //    chunks[max].getStripFromRight(*(dummy));
        //}
        //if (thisIndex < total - 1) {
        //    chunks[right].getStripFromLeft((A->access(myxdim, Range(0,myydim))));
        //} else {
        //    chunks[0].getStripFromLeft(*(dummy));
        //}

        //Array<double>* strip = new Array<double>(Domain<1>(Range(myydim)));
        if (thisIndex > 0) {
            for (int i = 0; i < myydim; i++) {
                (*(strip)).access(i) = (A->access(1, i));
            }
            chunks[thisIndex - 1].getStripFromRight(*(strip));
        } else {
            chunks[total - 1].getStripFromRight(*(strip));
        }
        if (thisIndex < total - 1) {
            for (int i = 0; i < myydim; i++) {
                (*(strip)).access(i) = (A->access(myxdim, i));
            }
            chunks[thisIndex + 1].getStripFromLeft(*(strip));
        } else {
            chunks[0].getStripFromLeft(*(strip));
        }
}

void Chunk::doStencil()
{
#ifdef RAW_STENCIL
    doStencil_raw();
    return;
#endif
    {
        double maxChange = 0.0;
        resetBoundary();
        if ((thisIndex != 0) && (thisIndex != total - 1)) {
            for (int i = 1; i < myxdim + 1; i++) {
                for (int j = 1; j < myydim - 1; j++) {
                    (*(B)).access(i, j) = (0.2) * ((*(A)).access(i, j) + (A->access(i, j + 1)) + (A->access(i, j - 1)) + (A->access(i + 1, j)) + (A->access(i - 1, j)));
                    double change = (*(B)).access(i, j) - (A->access(i, j));
                    if (change < 0) {
                        change = -change;
                    }
                    if (change > maxChange) {
                        maxChange = change;
                    }
                }
            }
        }
        if (thisIndex == 0) {
            for (int i = 1; i < myxdim; i++) {
                for (int j = 1; j < myydim - 1; j++) {
                    (*(B)).access(i, j) = (0.2) * ((*(A)).access(i, j) + (A->access(i, j + 1)) + (A->access(i, j - 1)) + (A->access(i + 1, j)) + (A->access(i - 1, j)));
                    double change = (*(B)).access(i, j) - (A->access(i, j));
                    if (change < 0) {
                        change = -change;
                    }
                    if (change > maxChange) {
                        maxChange = change;
                    }
                }
            }
        }
        if (thisIndex == total - 1) {
            for (int i = 1; i < myxdim; i++) {
                for (int j = 1; j < myydim - 1; j++) {
                    (*(B)).access(i, j) = (0.2) * ((*(A)).access(i, j) + (A->access(i, j + 1)) + (A->access(i, j - 1)) + (A->access(i + 1, j)) + (A->access(i - 1, j)));
                    double change = (*(B)).access(i, j) - (A->access(i, j));
                    if (change < 0) {
                        change = -change;
                    }
                    if (change > maxChange) {
                        maxChange = change;
                    }
                }
            }
        }
        Array<double, 2>* tmp = A;
        A = B;
        B = tmp;
    }
    #if _CHARJ_TRACE_ALL_METHODS
    traceUserBracketEvent(192321988, _charj_method_trace_timer, CkWallTimer());
    #endif
}

#define indexof(i,j,ydim) ( ((i)*(ydim)) + (j))
void Chunk::doStencil_raw()
{
    #if _CHARJ_TRACE_ALL_METHODS
    int _charj_method_trace_timer = CkWallTimer();
    #endif
    double* rA = A->raw();
    double* rB = B->raw();
    {
        double maxChange = 0.0;
        resetBoundary();

        if((thisIndex !=0)&&(thisIndex != total-1))
            for (int i=1; i<myxdim+1; i++)
                for (int j=1; j<myydim-1; j++) {
                    rB[indexof(i,j,myydim)] = 
                        (0.2)*(rA[indexof(i,  j,  myydim)] +
                                rA[indexof(i,  j+1,myydim)] +
                                rA[indexof(i,  j-1,myydim)] +
                                rA[indexof(i+1,j,  myydim)] +
                                rA[indexof(i-1,j,  myydim)]);

                    double change =  rB[indexof(i,j,myydim)] - rA[indexof(i,j,myydim)];
                    if (change < 0) change = - change;
                    if (change > maxChange) maxChange = change;
                }

        if(thisIndex == 0)
            for (int i=1; i<myxdim; i++)
                for (int j=1; j<myydim-1; j++) {
                    rB[indexof(i,j,myydim)] = 
                        (0.2)*(rA[indexof(i,  j,  myydim)] +
                                rA[indexof(i,  j+1,myydim)] +
                                rA[indexof(i,  j-1,myydim)] +
                                rA[indexof(i+1,j,  myydim)] +
                                rA[indexof(i-1,j,  myydim)]);

                    double change =  rB[indexof(i,j,myydim)] - rA[indexof(i,j,myydim)];
                    if (change < 0) change = - change;
                    if (change > maxChange) maxChange = change;
                }

        if(thisIndex == total-1) {
            for (int i=1; i<myxdim; i++)
                for (int j=1; j<myydim-1; j++) {
                    rB[indexof(i,j,myydim)] = 
                        (0.2)*(rA[indexof(i,  j,  myydim)] +
                                rA[indexof(i,  j+1,myydim)] +
                                rA[indexof(i,  j-1,myydim)] +
                                rA[indexof(i+1,j,  myydim)] +
                                rA[indexof(i-1,j,  myydim)]);

                    double change =  rB[indexof(i,j,myydim)] - rA[indexof(i,j,myydim)];
                    if (change < 0) change = - change;
                    if (change > maxChange) maxChange = change;
                }
        }
      
        Array<double, 2>* tmp = A;
        A = B;
        B = tmp;
    }
    #if _CHARJ_TRACE_ALL_METHODS
    traceUserBracketEvent(192321988, _charj_method_trace_timer, CkWallTimer());
    #endif
}


void Chunk::resetBoundary()
{
    #if _CHARJ_TRACE_ALL_METHODS
    int _charj_method_trace_timer = CkWallTimer();
    #endif
    {
        if (thisIndex != 0) {
            if (thisIndex < (int)(total / 2)) {
                for (int i = 1; i < myxdim + 1; i++) {
                    (*(A)).access(i, 0) = 1.0;
                }
            }
        }
        if (thisIndex == 0) {
            for (int i = 0; i < myxdim; i++) {
                (*(A)).access(i, 0) = 1.0;
            }
            for (int i = 0; 2 * i < myydim; i++) {
                (*(A)).access(0, i) = 1.0;
            }
        }
    }
    #if _CHARJ_TRACE_ALL_METHODS
    traceUserBracketEvent(480493615, _charj_method_trace_timer, CkWallTimer());
    #endif
}

void Chunk::processStripFromLeft(Array<double> __s)
{
    Array<double>* s = &__s;
    #if _CHARJ_TRACE_ALL_METHODS
    int _charj_method_trace_timer = CkWallTimer();
    #endif
    {
        if (thisIndex != 0) {
            for (int i = 0; i < myydim; i++) {
                (*(A)).access(0, i) = (s->access(i));
            }
        }
    }
    #if _CHARJ_TRACE_ALL_METHODS
    traceUserBracketEvent(186684624, _charj_method_trace_timer, CkWallTimer());
    #endif
}

void Chunk::processStripFromRight(Array<double> __s)
{
    Array<double>* s = &__s;
    #if _CHARJ_TRACE_ALL_METHODS
    int _charj_method_trace_timer = CkWallTimer();
    #endif
    {
        if (thisIndex != total - 1) {
            if (thisIndex != 0) {
                for (int i = 0; i < myydim; i++) {
                    (*(A)).access(myxdim + 1, i) = (s->access(i));
                }
            } else {
                for (int i = 0; i < myydim; i++) {
                    (*(A)).access(myxdim, i) = (s->access(i));
                }
            }
        }
    }
    #if _CHARJ_TRACE_ALL_METHODS
    traceUserBracketEvent(993349832, _charj_method_trace_timer, CkWallTimer());
    #endif
}

void Chunk::pup(PUP::er &p)
{
    A->pup(p);
    B->pup(p);
    p | myMax;
    p | myxdim;
    p | myydim;
    p | total;
    p | counter;
    __sdag_pup(p);
}

Chunk::Chunk()
{
    constructorHelper();
}
void Chunk::constructorHelper()
{
    __sdag_init();
}

Chunk::Chunk(CkMigrateMessage *m)
{
    constructorHelper();
}
bool Chunk::_trace_registered = false;
void Chunk::_initTrace() {
    #if _CHARJ_TRACE_ALL_METHODS || _CHARJ_TRACE_TRACED_METHODS
    if (_trace_registered) return;
    traceRegisterUserEvent("Chunk.Chunk", 43910155);
    traceRegisterUserEvent("Chunk.sendStrips", 253859300);
    traceRegisterUserEvent("Chunk.doStencil", 192321988);
    traceRegisterUserEvent("Chunk.resetBoundary", 480493615);
    traceRegisterUserEvent("Chunk.processStripFromLeft", 186684624);
    traceRegisterUserEvent("Chunk.processStripFromRight", 993349832);
    traceRegisterUserEvent("Chunk.jacobi", 572453224);
    _trace_registered = true;
    #endif
}


//#include "Chunk.def.h"


#include "jacobi.def.h"
