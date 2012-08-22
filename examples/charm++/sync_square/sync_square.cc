#include "sync_square.h"
#include <stdlib.h>

Driver::Driver(CkArgMsg* args) {
    int value = 10;
    if (args->argc > 1) value = strtol(args->argv[1], NULL, 10);
    delete args;
    s = CProxy_Squarer::ckNew();
    thisProxy.get_square(value);
}

void Driver::get_square(int value)
{
    int_message* square_message = s.square(value);
    int square = square_message->value;
    CkFreeMsg(square_message);
    CkExit();
}

int_message* Squarer::square(int x) {
    return new int_message(x*x);
}


#include "sync_square.def.h"
