#ifndef CK_IO_H
#define CK_IO_H

#include "CkIO.decl.h"

namespace Ck { namespace IO {
  /// Identifier for a file to be accessed
  typedef int Token;

  class Options {

  };

  /// Class to mediate IO operations between Charm++ application code
  /// and underlying filesystems.
  ///
  /// Tokens are passed to @arg ready callbacks, which the application
  /// then passes to the local methods when initiating operations.
  class Manager : public CBase_Manager {
  public:
    Manager();

    void prepareOutput(const char *name, size_t bytes,
		       CkCallback ready, CkCallback complete,
		       Options opts = Options());
    void write(Token token, const void *data, size_t bytes, size_t offset);

    void prepareInput(const char *name, CkCallback ready,
		      Options opts = Options());
    void read(Token token, void *data, size_t bytes, size_t offset,
	      CkCallback complete);
  };

  }}
#endif
