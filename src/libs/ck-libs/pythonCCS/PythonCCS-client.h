/* Security note:

   The dynamic insertion of a code into a running program can generate bugs, and
   potentially lead the parallel program to die. Moreover, a malicious client
   can send a handcrafted message which corrupts the status of the server (i.e.
   it can delete or modify the live data, or mess up with the interpreter
   assigned to another user.

   The current scheme does not try to resolve all conflicts and rely on the
   correctness of the client to act honesly. In particular, the programmer
   building the client should take care that if the current file is not included
   (i.e. client build in a language different from c/c++) the class structure is
   coherent with this file. Also the definition of the class inheriting from
   PythonIterator needs to be coherent across client and server.
*/

#include "converse.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* This class is empty, and should be reimplemented by the user */
class PythonIterator {
  /* _It is strongly suggested to avoid pointers_, in which case none of the
     following methods needs to be reimplemented. Instead, if pointers are used,
     the following three methods have to be reimplemented to pack and unpack the
     pointered values across client and server (in the same semantics of charm
     packing)
*/
 public:
  // return the total size of the object (after packing)
  virtual int size() {return sizeof(*this);};

  // pack the message into a contiguous memory and return the pointer to this
  // memory. the memory needs to be persistent (i.e. cannot disappear) and will
  // be freed automatically
  virtual PythonIterator *pack() {
    void *memory = malloc(size());
    memcpy (memory, this, size());
    return (PythonIterator *)memory;
  };

  // unpack needs to reposition the pointer in the case they are used
  virtual void unpack() {};
};

/* This class contains a single unsigned 4 bytes integer, it is needed in order
   to distinguish between PythonExecute and PythonPrint on the server side,
   since a byte copy does not preserve class information.
*/
class PythonAbstract {
 public:
  /* magic contains the size of the class passed, it is filled by the
     constructors of the inheriting classes */
  CmiUInt4 magic;
};

class PythonExecute : private PythonAbstract {
  friend class PythonObject;
 private:
  int codeLength;
  char * code;

  /* the following parameters are used when the iterator mode is invoked */
  int methodNameLength;
  char * methodName;
  int infoSize; /* must contain the size of the info structure passed */
  PythonIterator *info;

  char interpreter[4]; /* request for an existing interpreter */
  char flags;
  /* flags has the following parameters: (bit 1 is the MSB)
     bit 1: isPersistent
     bit 2: keepPrint
     bit 3: isHighLevel
     bit 4: isIterate
     bit 5: print request
  */
  static const char FLAG_PERSISTENT = 0x80;
  static const char FLAG_KEEPPRINT = 0x40;
  static const char FLAG_HIGHLEVEL = 0x20;
  static const char FLAG_ITERATE = 0x10;
 public:

  /* constructors */
  /* by default, if the code is persistent, then the prints will be maintained */
  PythonExecute(char *_code, bool _persistent=false, bool _highlevel=false, char _interp[4]=0);
  PythonExecute(char *_code, char *_method, PythonIterator *_info, bool _persistent=false, bool _highlevel=false, char _interp[4]=0);
  ~PythonExecute();

  /* routines to set all parameters in the class */
  void setCode(char *_set);
  void setMethodName(char *_set);
  void setIterator(PythonIterator *_set);
  void setPersistent(bool _set);
  void setIterate(bool _set);
  void setHighLevel(bool _set);
  void setKeepPrint(bool _set);
  void setInterpreter(char i[4]) { memcpy(interpreter, i, 4); };

  bool isPersistent() { return flags & FLAG_PERSISTENT; };
  bool isIterate() { return flags & FLAG_ITERATE; };
  bool isHighLevel() { return flags & FLAG_HIGHLEVEL; };
  bool isKeepPrint() { return flags & FLAG_KEEPPRINT; };
  void getInterpreter(char i[4]) { memcpy(i, interpreter, 4); };

  int size();
  char *toString();
  void unpack();
};

class PythonPrint : private PythonAbstract {
  friend class PythonObject;
 private:
  char interpreter[4];;
  char flags;
  /* flags has the following parameters: (bit 1 is the MSB)
     bit 1: noWait
  */
  static const char FLAG_WAIT = 0x80;
 public:
  PythonPrint(char _interp[4], bool Wait=true);

  void setWait(bool _set);
  bool isWait() { return flags & FLAG_WAIT; };
};
