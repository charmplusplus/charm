#ifndef CK_IO_H
#define CK_IO_H

#include <vector>
#include <string>
#include <pup.h>
#include <ckcallback.h>

#include "CkIO.decl.h"

namespace Ck { namespace IO {
  /// Note: The values in options are not currently a stable or working interface.
  /// Users should not set anything in them.
  struct Options {
    Options()
      : peStripe(0), writeStripe(0), activePEs(-1), basePE(-1), skipPEs(-1), read_stripe(0)
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
    // How many bytes each Read Session should hold
    size_t read_stripe;

    void pup(PUP::er &p) {
      p|peStripe;
      p|writeStripe;
      p|activePEs;
      p|basePE;
      p|skipPEs;
      p|read_stripe;
    }
  };

  class File;
  class Session;

  /// Open the named file on the selected subset of PEs, and send a
  /// FileReadyMsg to the opened callback when the system is ready to accept
  /// session requests on that file.
  /// Note: The values in options are not currently a stable or working interface.
  /// Users should not set anything in them.
  void open(std::string name, CkCallback opened, Options opts);

  /// Prepare to write data into the file described by token, in the window
  /// defined by the offset and byte length. When the session is set up, a
  /// SessionReadyMsg will be sent to the ready callback. When all of the data
  /// has been written and synced, a message will be sent to the complete
  /// callback.
  void startSession(File file, size_t bytes, size_t offset,
                    CkCallback ready, CkCallback complete);

  /// Prepare to write data into @arg file, in the window defined by the @arg
  /// offset and length in @arg bytes. When the session is set up, a
  /// SessionReadyMsg will be sent to the @arg ready callback. When all of the
  /// data has been written and synced, an additional write will be made to the
  /// file to `commit' the session's work. When that write has completed, a
  /// message will be sent to the @arg complete callback.
  void startSession(File file, size_t bytes, size_t offset, CkCallback ready,
                    const char *commitData, size_t commitBytes, size_t commitOffset,
                    CkCallback complete);

  /// Write the given data into the file to which session is attached. The
  /// offset is relative to the file as a whole, not to the session's offset.
  void write(Session session, const char *data, size_t bytes, size_t offset);

  /// Close a previously-opened file. All sessions on that file must have
  /// already signalled that they are complete.
  void close(File file, CkCallback closed);
  
  /**
   * Prepare to read data from @arg file section specified by @arg bytes and @arg offset.
   * This method will proceed to eagerly read all of the data in that window into memory
   * for future read calls. After all the data is read in, the ready callback will be invoked.
   * The ready callback will take in a SessionReadyMessage* that will contain the offset, the amount of bytes
   * , and the buffer in the form of a vector<char>.
   */
  void startReadSession(File file, size_t bytes, size_t offset, CkCallback ready);
  /**
   * Is a method that reads data from the @arg session of length @arg bytes at offset
   * @arg offset. After this read finishes, the @arg after_read callback is invoked, taking 
   * a ReadCompleteMsg* which points to a vector<char> buffer, the offset, and the number of 
   * bytes of the read.
   * */
  void read(Session session, size_t bytes, size_t offset, CkCallback after_read); 

  class File {
    int token;
    friend void startSession(File file, size_t bytes, size_t offset,
                             CkCallback ready, CkCallback complete);

    friend void startReadSession(File file, size_t bytes, size_t offset, CkCallback ready);

    friend void startSession(File file, size_t bytes, size_t offset, CkCallback ready,
                             const char *commitData, size_t commitBytes, size_t commitOffset,
                             CkCallback complete);
    friend void close(File file, CkCallback closed);
    friend class FileReadyMsg;

  public:
    File(int token_) : token(token_) { }
    File() : token(-1) { }
    void pup(PUP::er &p) { p|token; }
  };

  class FileReadyMsg : public CMessage_FileReadyMsg {
  public:
    File file;
    FileReadyMsg(const File &tok) : file(tok) {}
  };

  namespace impl { class Manager; }

  class Session {
    int file;
    size_t bytes, offset;
    CkArrayID sessionID;
    friend class Ck::IO::impl::Manager;
  public:
    Session(int file_, size_t bytes_, size_t offset_,
            CkArrayID sessionID_)
      : file(file_), bytes(bytes_), offset(offset_), sessionID(sessionID_)
      { }
    Session() { }
    void pup(PUP::er &p) {
      p|file;
      p|bytes;
      p|offset;
      p|sessionID;
    }
  };

  class SessionReadyMsg : public CMessage_SessionReadyMsg {
  public:
    Session session;
    SessionReadyMsg(Session session_) : session(session_) { }
  };

  class ReadCompleteMsg : public CMessage_ReadCompleteMsg {
	public:
	    std::vector<char> data;
	    size_t offset;
	    size_t bytes;

	    ReadCompleteMsg(std::vector<char> buffer, size_t in_offset, size_t in_bytes) : data(buffer), offset(in_offset), bytes(in_bytes){

	    }
		

  };



}}
#endif
