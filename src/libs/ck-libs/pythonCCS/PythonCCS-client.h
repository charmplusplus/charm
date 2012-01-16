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
#include <netinet/in.h>

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
  // be freed automatically. It should not modify the original data.
  virtual char *pack() {
    void *memory = malloc(size());
    memcpy (memory, this, size());
    return (char *)memory;
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

  bool isFinished();
  bool isExecute();
  bool isPrint();
  void unpack();
};

/* the unions with the "dummy" variables are to force a 64 bit space, which will
   allow compatibility with 64 bit machines */
/* all the integers (CmiUInt4) are interpreted in network byte order */
class PythonExecute : private PythonAbstract {
  friend class PythonObject;
  friend class CpdPythonGroup;
 private:
  CmiUInt4 codeLength;
  union {
    char * code;
    char dummy[8];
  } code;

  /* the following parameters are used when the iterator mode is invoked */
  union {
    char * methodName;
    char dummy[8];
  } methodName;
  CmiUInt4 methodNameLength;
  CmiUInt4 infoSize; /* must contain the size of the info structure passed */
  union {
    PythonIterator *info;
    char dummy[8];
  } info;

  /* interpreter does not necessarely needs to be sent in byte order (since the
     client does not need to read it), but it is to have a common interface with
     java */
  CmiUInt4 interpreter; /* request for an existing interpreter */
  unsigned char flags;
  /* flags has the following parameters: (bit 1 is the MSB)
     bit 1: isPersistent
     bit 2: keepPrint
     bit 3: isHighLevel
     bit 4: isIterate
  */
  static const unsigned char FLAG_PERSISTENT = 0x80;
  static const unsigned char FLAG_KEEPPRINT = 0x40;
  static const unsigned char FLAG_HIGHLEVEL = 0x20;
  static const unsigned char FLAG_ITERATE = 0x10;
  static const unsigned char FLAG_WAIT = 0x08;
  static const unsigned char FLAG_NOCHECK = 0x04;
 public:
  static const CmiUInt4 localmagic = 37492037;

  /* constructors */
  /* by default, if the code is persistent, then the prints will be maintained */
  PythonExecute(char *_code, bool _persistent=false, bool _highlevel=false, CmiUInt4 _interp=0);
  PythonExecute(char *_code, char *_method, PythonIterator *_info, bool _persistent=false, bool _highlevel=false, CmiUInt4 _interp=0);
  ~PythonExecute();

  /* routines to set all parameters in the class */
  void setCode(char *_set);
  void setMethodName(char *_set);
  void setIterator(PythonIterator *_set);
  void setPersistent(bool _set);
  void setIterate(bool _set);
  void setHighLevel(bool _set);
  void setKeepPrint(bool _set);
  void setWait(bool _set);
  void setInterpreter(CmiUInt4 i) { interpreter = i; };
  void setNoCheck(bool _set);

  bool isPersistent() { return flags & FLAG_PERSISTENT; };
  bool isIterate() { return flags & FLAG_ITERATE; };
  bool isHighLevel() { return flags & FLAG_HIGHLEVEL; };
  bool isKeepPrint() { return flags & FLAG_KEEPPRINT; };
  bool isWait() { return flags & FLAG_WAIT; };
  CmiUInt4 getInterpreter() { return interpreter; };
  bool isNoCheck() { return flags & FLAG_NOCHECK; };

  int size();
  char *pack();
  void unpack();

  void print(); /* for debugging */
};

class PythonPrint : private PythonAbstract {
  friend class PythonObject;
 private:
  CmiUInt4 interpreter;
  unsigned char flags;
  /* flags has the following parameters: (bit 1 is the MSB)
     bit 1: noWait
     bit 2: kill previous request
  */
  static const unsigned char FLAG_WAIT = 0x80;
  static const unsigned char FLAG_KILL = 0x40;
 public:
  static const CmiUInt4 localmagic = 989370215;
  PythonPrint(CmiUInt4 _interp, bool Wait=true, bool Kill=false);

  void setWait(bool _set);
  bool isWait() { return flags & FLAG_WAIT; };

  void setKill(bool _set);
  bool isKill() { return flags & FLAG_KILL; };


  int size() { return sizeof(*this); };
  char *pack() { magic = htonl(magic); //interpreter = htonl(interpreter);
 return (char *)this; };
  void unpack() { interpreter = ntohl(interpreter); }

  void print(); /* for debugging */
};

class PythonFinished : private PythonAbstract {
  friend class PythonObject;
 private:
  CmiUInt4 interpreter;
  unsigned char flags;
  /* flags has the following parameters: (bit 1 is the MSB)
     bit 1: noWait
  */
  static const unsigned char FLAG_WAIT = 0x80;
 public:
  static const CmiUInt4 localmagic = 738963580;
  PythonFinished(CmiUInt4 _interp, bool Wait=true);

  void setWait(bool _set);
  bool isWait() { return flags & FLAG_WAIT; };

  int size() { return sizeof(*this); };
  char *pack() { magic = htonl(magic); //interpreter = htonl(interpreter);
 return (char *)this; };
  void unpack() { interpreter = ntohl(interpreter); }

  void print(); /* for debugging */
};
