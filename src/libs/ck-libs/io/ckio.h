#ifndef CK_IO_H
#define CK_IO_H

#include <string>
#include <pup_stl.h>

namespace Ck { namespace IO {
  class FileReadyMsg;
  class SessionReadyMsg;

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

  /// Open the named file on the selected subset of PEs, and send a
  /// FileReadyMsg to the opened callback when the system is ready to accept
  /// session requests on that file.
  void open(std::string name, CkCallback opened, Options opts);

  /// Prepare to write data into the file described by token, in the window
  /// defined by the offset and byte length. When the session is set up, a
  /// SessionReadyMsg will be sent to the ready callback. When all of the data
  /// has been written and synced, a message will be sent to the complete
  /// callback.
  void startSession(FileReadyMsg *file, size_t bytes, size_t offset,
                    CkCallback ready, CkCallback complete);

  /// Write the given data into the file to which session is attached. The
  /// offset is relative to the file as a whole, not to the session's offset.
  void write(SessionReadyMsg *session, const char *data, size_t bytes, size_t offset);

  /// Close a previously-opened file. All sessions on that file must have
  /// already signalled that they are complete.
  void close(FileReadyMsg *file, CkCallback closed);
}}
#endif
