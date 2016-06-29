#include "sync_square.h"
#include <stdlib.h>

Driver::Driver(CkArgMsg* args) {
    int value = 10;
    if (args->argc > 1) value = strtol(args->argv[1], NULL, 10);
    delete args;
    s = CProxy_Squarer::ckNew();
    sarr = CProxy_SquarerArr::ckNew(2);
    sgrp = CProxy_SquarerGrp::ckNew();
    thisProxy.get_square(value);
}

void Driver::get_square(int val)
{
    const char* container[3] = {"chare", "1d Array", "Group"}; 
    int square;
    int_message* square_msg;
    for(int i=0; i<3; i++){
      CkPrintf("%s: \n", container[i]);
      square = (i==0? s.square(val): (i==1? sarr[0].square(val): sgrp[0].square(val)));  
      CkAssert(square == val*val);
      CkPrintf(" %d^2 = %d\n", val, square);
      square_msg = (i==0? s.squareM(val): (i==1? sarr[0].squareM(val): sgrp[0].squareM(val)));
      square = square_msg->value;
      CkAssert(square == val*val);
      CkPrintf(" %d^2 = %d (using Msg)\n", val, square);
      CkFreeMsg(square_msg);
    }
    CkExit();
}

int Squarer::square(int x) {
    return (x*x);
}

int_message* Squarer::squareM(int x) {
    return new int_message (x*x);
}


int SquarerArr::square(int x) {
    return (x*x);
}

int_message* SquarerArr::squareM(int x) {
    return new int_message (x*x);
}


int SquarerGrp::square(int x) {
    return (x*x);
}

int_message* SquarerGrp::squareM(int x) {
    return new int_message (x*x);
}
#include "sync_square.def.h"
