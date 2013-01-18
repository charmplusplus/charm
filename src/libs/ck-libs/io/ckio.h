#ifndef CK_IO_H
#define CK_IO_H

#include <cstring>
#include <string>
#include <algorithm>
#include <utility>
#include <fcntl.h>
#include <pup_stl.h>

namespace Ck { namespace IO {
  /// Identifier for a file to be accessed
  typedef int Token;

  struct Options {
    Options()
      : peStripe(-1), writeStripe(-1), activePEs(-1), basePE(-1), skipPEs(-1)
      { }

    /// How much contiguous data (in bytes) should be assigned to each active PE
    size_t peStripe;
    /// How much contiguous data (in bytes) should a PE gather before writing it out
    size_t writeStripe;
    /// How many PEs should participate in this activity
    int activePEs;
    /// Which PE should be the first to participate in this activity
    int basePE;
    /// How should active PEs be spaced out?
    int skipPEs;

    void pup(PUP::er &p) {
      p|peStripe;
      p|writeStripe;
      p|activePEs;
      p|basePE;
      p|skipPEs;
    }
  };

    struct FileReadyMsg;
  }
}

#include "CkIO.decl.h"
#include <map>
#include <vector>

namespace Ck { namespace IO {
  struct FileReadyMsg : public CMessage_FileReadyMsg {
    Token token;
    FileReadyMsg(const Token &tok) : token(tok) {}
  };
  
  struct buffer
  {
    std::vector<char> array;
    int bytes_filled_so_far;
    
    buffer()
    {
      bytes_filled_so_far = 0;
    }

    void expect(size_t bytes)
    {
      array.resize(bytes);
    }
    
    void insertData(const char *data, size_t length, size_t offset)
    {
      char *dest = &array[offset];
      memcpy(dest, data, length);

      bytes_filled_so_far += length;
    }

    bool isFull()
    {
      return bytes_filled_so_far == array.size();
    }
  };
    

  struct FileInfo {
    std::string name;
    Options opts;
    size_t bytes, total_written;
    int fd;
    CkCallback complete;
    std::map<size_t, struct buffer> bufferMap;

    FileInfo(std::string name_, size_t bytes_, Options opts_)
    : name(name_), opts(opts_), bytes(bytes_), total_written(0), fd(-1)
      { }
    FileInfo()
    : bytes(-1), total_written(-1), fd(-1)
      { }
  };

  /// Class to mediate IO operations between Charm++ application code
  /// and underlying filesystems.
  ///
  /// Tokens are passed to @arg ready callbacks, which the application
  /// then passes to the local methods when initiating operations.
  class Manager : public CBase_Manager {
  public:
    Manager();

    Manager_SDAG_CODE;

    /// Application-facing methods, invoked locally on the calling PE
    void prepareOutput(const char *name, size_t bytes,
		       CkCallback ready, CkCallback complete,
		       Options opts = Options());
    void write(Token token, const char *data, size_t bytes, size_t offset);

    void prepareInput(const char *name, CkCallback ready,
		      Options opts = Options());
    void read(Token token, void *data, size_t bytes, size_t offset,
	      CkCallback complete);


    /// Internal methods, used for interaction among IO managers across the system
    void write_forwardData(Token token, const char *data, size_t bytes, size_t offset);
    void write_dataWritten(Token token, size_t bytes);

  private:
    int filesOpened;
    Token nextToken;
    std::map<Token, FileInfo> files;
    CkCallback nextReady;

    int lastActivePE(const Options &opts) {
      return opts.basePE + (opts.activePEs-1)*opts.skipPEs;
    }
    int openFile(const std::string& name);
  };

  }}
#endif
