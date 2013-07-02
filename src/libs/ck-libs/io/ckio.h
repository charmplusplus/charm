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
  typedef int FileToken;
  typedef int SessionToken;

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
    FileToken token;
    FileReadyMsg(const FileToken &tok) : token(tok) {}
  };

  namespace impl {  
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
    
    struct SessionInfo {
      FileToken file;
      size_t bytes, offset, total_written;
      int pesReady;
      CkCallback complete;
      std::map<size_t, struct buffer> bufferMap;

      SessionInfo(FileToken file_, size_t bytes_, size_t offset_, CkCallback complete_)
        : file(file_), bytes(bytes_), offset(offset_), complete(complete_)
        { }
      SessionInfo()
        : file(-1)
    };

    struct FileInfo {
      std::string name;
      Options opts;
      int fd;

      FileInfo(std::string name_, Options opts_)
        : name(name_), opts(opts_), fd(-1)
        { }
      FileInfo()
        : fd(-1)
        { }
    };
  }

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
    void openWrite(std::string name, CkCallback opened, Options opts = Options());
    void prepareWrite(size_t bytes, size_t offset, CkCallback ready, CkCallback complete);
    void write(FileToken file, SessionToken session,
               const char *data, size_t bytes, size_t offset);

#if 0
    void prepareInput(const char *name, CkCallback ready,
		      Options opts = Options());
    void read(Token token, void *data, size_t bytes, size_t offset,
	      CkCallback complete);
#endif

    /// Internal methods, used for interaction among IO managers across the system
    void write_forwardData(SessionToken token, const char *data, size_t bytes, size_t offset);
    void write_dataWritten(SessionToken token, size_t bytes);

  private:
    int filesOpened, sessionsOpened;
    FileToken nextToken;
    std::map<FileToken, impl::FileInfo> files;
    std::map<SessionToken, impl::SessionInfo> sessions;

    CkCallback nextReady;

    int lastActivePE(const Options &opts) {
      return opts.basePE + (opts.activePEs-1)*opts.skipPEs;
    }
    int openFile(const std::string& name);
  };

  }}
#endif
