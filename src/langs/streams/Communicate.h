#ifndef COMMUNICATE_H
#define COMMUNICATE_H

extern "C" {
#include "converse.h"
}

class MIStream;
class MOStream;

#define ALL      -1
#define ALLBUTME -2
#define BUFSIZE  4096
#define ANY      -1

class Communicate {

private:
  int CsmHandlerIndex;

public:
  Communicate(void);
  ~Communicate();
  MIStream *newInputStream(int pe, int tag);
  MOStream *newOutputStream(int pe, int tag, unsigned int bufsize);
  void *getMessage(int PE, int tag);
  void sendMessage(int PE, void *msg, int size);
};

#include "MStream.h"

#endif
