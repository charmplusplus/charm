#ifndef CK_IO_H
#define CK_IO_H

#include <string>
#include <pup_stl.h>

namespace Ck { namespace IO {
  /// Identifier for a file to be accessed
  typedef int FileToken;
  class SessionReadyMessage;

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

  void open(std::string name, CkCallback opened, Options opts);
  void startSession(FileToken token, size_t bytes, size_t offset,
                    CkCallback ready, CkCallback complete);
  void write(SessionReadyMessage *session, const char *data, size_t bytes, size_t offset);
  void close(FileToken token, CkCallback closed);

  struct FileReadyMsg : public CMessage_FileReadyMsg {
    FileToken token;
    FileReadyMsg(const FileToken &tok) : token(tok) {}
  };
}}
#endif
