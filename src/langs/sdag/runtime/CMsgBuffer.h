/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _CMsgBufer_H_
#define _CMsgBufer_H_

class CMsgBuffer {
  public:
    int entry;
    void *msg;
    int refnum;
    CMsgBuffer *next;
    CMsgBuffer(int e, void *m, int r) : entry(e), msg(m), refnum(r) {}
};

#endif
