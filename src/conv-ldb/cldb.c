#include "converse.h"
/*
 * How to write a load-balancer:
 *
 * 1. Every load-balancer must contain a definition of struct CldField.
 *    This structure describes what kind of data will be piggybacked on
 *    the messages that go through the load balancer.  The structure
 *    must include certain predefined fields.  Put the word
 *    CLD_STANDARD_FIELD_STUFF at the front of the struct definition
 *    to include these predefined fields.
 *
 * 2. Every load-balancer must contain a definition of the global variable
 *    Cld_fieldsize.  You must initialize this to sizeof(struct CldField).
 *    This is not a CPV or CSV variable, it's a plain old C global.
 *
 * 3. When you send a message, you'll probably want to temporarily
 *    switch the handler.  The following function will switch the handler
 *    while saving the old one:
 *
 *       CldSwitchHandler(msg, field, newhandler);
 *
 *    Field must be a pointer to the gap in the message where the CldField
 *    is to be stored.  The switch routine will use this region to store
 *    the old handler, as well as some other stuff.  When the message
 *    gets handled, you can switch the handler back like this:
 *    
 *       CldRestoreHandler(msg, &field);
 *
 *    This will not only restore the handler, it will also tell you
 *    where in the message the CldField was stored.
 *
 * 4. Don't forget that CldEnqueue must support directed transmission of
 *    messages as well as undirected, and broadcasts too.
 *
 */

int CldRegisterInfoFn(CldInfoFn fn)
{
  return CmiRegisterHandler((CmiHandler)fn);
}

int CldRegisterPackFn(CldPackFn fn)
{
  return CmiRegisterHandler((CmiHandler)fn);
}

/*
 * These next subroutines are balanced on a thin wire.  They're
 * correct, but the slightest disturbance in the offsets could break them.
 *
 */

void CldSwitchHandler(char *cmsg, int *field, int handler)
{
  int *data = (int*)(cmsg+CmiMsgHeaderSizeBytes);
  field[1] = CmiGetHandler(cmsg);
  field[0] = data[0];
  data[0] = ((char*)field)-cmsg;
  CmiSetHandler(cmsg, handler);
}

void CldRestoreHandler(char *cmsg, void *hfield)
{
  int *data = (int*)(cmsg+CmiMsgHeaderSizeBytes);
  int offs = data[0];
  int *field = (int*)(cmsg+offs);
  data[0] = field[0];
  CmiSetHandler(cmsg, field[1]);
  *(int**)hfield = field;
}

