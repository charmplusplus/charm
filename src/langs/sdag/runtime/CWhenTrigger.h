#ifndef _CWhenTrigger_H_
#define _CWhenTrigger_H_

#include <stddef.h> // for size_t

#define MAXARG 8
#define MAXANY 8
#define MAXREF 8

class CWhenTrigger {
  public:
    int whenID, nArgs;
    size_t args[MAXARG];
    int nAnyEntries;
    int anyEntries[MAXANY];
    int nEntries;
    int entries[MAXREF];
    int refnums[MAXREF];
    CWhenTrigger *next;
    CWhenTrigger(int id, int na, int ne, int nae) :
       whenID(id), nArgs(na), nEntries(ne), nAnyEntries(nae){}
};
#endif
