#include <string>
#include <map>
#include <algorithm>
#include <sstream>

typedef int FileToken;
#include "CkIO.decl.h"
#include "CkIO_impl.decl.h"

#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <pup_stl.h>

#if defined(_WIN32)
#include <io.h>
#endif

using std::min;
using std::max;
using std::map;
using std::string;

namespace Ck { namespace IO {
    namespace impl {
      CProxy_Director director;
      CkpvDeclare(Manager *, manager);
    }


    namespace impl {
      struct FileInfo {
        string name;
        CkCallback opened;
        Options opts;
        int fd;
        CProxy_WriteSession session;
        CkCallback complete;

        FileInfo(string name_, CkCallback opened_, Options opts_)
          : name(name_), opened(opened_), opts(opts_), fd(-1)
        { }
        FileInfo(string name_, Options opts_)
          : name(name_), opened(), opts(opts_), fd(-1)
        { }
        FileInfo()
          : fd(-1)
        { }
      };

      void fatalError(string desc, string file) {
        std::stringstream out;
        out << "FATAL ERROR on PE " << CkMyPe()
            << " working on file '" << file << "': "
            << desc << "; system reported " << strerror(errno) << std::endl;
        CkAbort(out.str().c_str());
      }

      class Director : public CBase_Director {
        int filesOpened;
        map<FileToken, impl::FileInfo> files;
        CProxy_Manager managers;
        int opnum, sessionID;
        Director_SDAG_CODE

      public:
        Director(CkArgMsg *m)
          : filesOpened(0), opnum(0), sessionID(0)
        {
          delete m;
          director = thisProxy;
          managers = CProxy_Manager::ckNew();
        }

        Director(CkMigrateMessage *m) : CBase_Director(m) { }

        void pup(PUP::er &p) {
          // FIXME: All files must be closed across checkpoint/restart
          if (files.size() != 0)
            CkAbort("CkIO: All files must be closed across checkpoint/restart");

          p | filesOpened;
          p | managers;
          p | opnum;
          p | sessionID;
        }

        void openFile(string name, CkCallback opened, Options opts) {
          if (0 == opts.peStripe)
            opts.peStripe = 16 * 1024 * 1024;
          if (0 == opts.writeStripe)
            opts.writeStripe = 4 * 1024 * 1024;
          if (-1 == opts.activePEs)
            opts.activePEs = min(CkNumPes(), 32);
          if (-1 == opts.basePE)
            opts.basePE = 0;
          if (-1 == opts.skipPEs)
            opts.skipPEs = CkMyNodeSize();

          files[filesOpened] = FileInfo(name, opened, opts);
          managers.openFile(opnum++, filesOpened++, name, opts);
        }

        void fileOpened(FileToken file) {
          files[file].opened.send(new FileReadyMsg(file));
        }

        void prepareWriteSession(FileToken file, size_t bytes, size_t offset,
                                 CkCallback ready, CkCallback complete) {
          Options &opts = files[file].opts;

	  int numStripes = 0;
	  size_t bytesLeft = bytes, delta = opts.peStripe - offset % opts.peStripe;
	  // Align to stripe boundary
	  if (offset % opts.peStripe != 0 && delta < bytesLeft) {
	    bytesLeft -= delta;
	    numStripes++;
	  }
	  numStripes += bytesLeft / opts.peStripe;
	  if (bytesLeft % opts.peStripe != 0)
	    numStripes++;

          CkArrayOptions sessionOpts(numStripes);
          //sessionOpts.setMap(managers);
          files[file].session =
            CProxy_WriteSession::ckNew(file, offset, bytes, sessionOpts);
          CkAssert(files[file].complete.isInvalid());
          files[file].complete = complete;
          ready.send(new SessionReadyMsg(Session(file, bytes, offset,
                                                 files[file].session)));
        }

        void sessionComplete(FileToken token) {
          CProxy_CkArray(files[token].session.ckGetArrayID()).ckDestroy();
          files[token].complete.send(CkReductionMsg::buildNew(0, NULL));
          files[token].complete = CkCallback(CkCallback::invalid);
        }

        void close(FileToken token, CkCallback closed) {
          managers.close(opnum++, token, closed);
          files.erase(token);
        }
      };

      class Manager : public CBase_Manager {
        Manager_SDAG_CODE
        int opnum;

      public:
        Manager()
          : opnum(0)
        {
          CkpvInitialize(Manager*, manager);
          CkpvAccess(manager) = this;
          thisProxy[CkMyPe()].run();
        }

        Manager(CkMigrateMessage *m)
          : CBase_Manager(m)
        {
          CkpvInitialize(Manager*, manager);
          CkpvAccess(manager) = this;
        }

        void pup(PUP::er &p) {
          p | opnum;

          // FIXME: All files must be closed across checkpoint/restart
          if (files.size() != 0)
            CkAbort("CkIO: All files must be closed across checkpoint/restart");
        }

        void prepareFile(FileToken token, string name, Options opts) {
          CkAssert(files.end() == files.find(token));
          //CkAssert(lastActivePE(opts) < CkNumPes());
          CkAssert(opts.writeStripe <= opts.peStripe);
          files[token] = impl::FileInfo(name, opts);

          contribute(sizeof(FileToken), &token, CkReduction::max_int,
                     CkCallback(CkReductionTarget(Director, fileOpened), director));
        }

        impl::FileInfo* get(FileToken token) {
          CkAssert(files.find(token) != files.end());

          // Open file if we're one of the active PEs
          // XXX: Or maybe wait until the first write-out, to smooth the metadata load?
          if (files[token].fd == -1) {
            string& name = files[token].name;
#if defined(_WIN32)
            int fd = CmiOpen(name.c_str(), _O_WRONLY | _O_CREAT, _S_IREAD | _S_IWRITE);
#else
            int fd = CmiOpen(name.c_str(), O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR);
#endif
            if (-1 == fd)
              fatalError("Failed to open a file for parallel output", name);

            files[token].fd = fd;
          }

          return &(files[token]);
        }

        void write(Session session, const char *data, size_t bytes, size_t offset) {
          Options &opts = files[session.file].opts;
          size_t stripe = opts.peStripe;

          CkAssert(offset >= session.offset);
          CkAssert(offset + bytes <= session.offset + session.bytes);

          size_t sessionStripeBase = (session.offset / stripe) * stripe;

          while (bytes > 0) {
            size_t stripeIndex = (offset - sessionStripeBase) / stripe;
            size_t bytesToSend = min(bytes, stripe - offset % stripe);

            CProxy_WriteSession(session.sessionID)[stripeIndex]
              .forwardData(data, bytesToSend, offset);

            data += bytesToSend;
            offset += bytesToSend;
            bytes -= bytesToSend;
          }
        }

        void doClose(FileToken token, CkCallback closed) {
          int fd = files[token].fd;
          if (fd != -1) {
            int ret;
            do {
#if defined(_WIN32)
              ret = _close(fd);
#else
              ret = ::close(fd);
#endif
            } while (ret < 0 && errno == EINTR);
            if (ret < 0)
              fatalError("close failed", files[token].name);
          }
          files.erase(token);
          contribute(closed);
        }

        int procNum(int arrayHdl,const CkArrayIndex &element)
        {
#if 0
          int peIndex = stripeIndex % opts.activePEs;
          int pe = opts.basePE + peIndex * opts.skipPEs;
#endif
          return 0;
        }

      private:
        map<FileToken, impl::FileInfo> files;

        int lastActivePE(const Options &opts) {
          return opts.basePE + (opts.activePEs-1)*opts.skipPEs;
        }
      };

      class WriteSession : public CBase_WriteSession {
        const FileInfo *file;
        size_t sessionOffset, myOffset;
        size_t sessionBytes, myBytes, myBytesWritten;
        FileToken token;

        struct buffer {
          std::vector<char> array;
          int bytes_filled_so_far;

          buffer() {
            bytes_filled_so_far = 0;
          }

          void expect(size_t bytes) {
            array.resize(bytes);
          }

          void insertData(const char *data, size_t length, size_t offset) {
            char *dest = &array[offset];
            memcpy(dest, data, length);

            bytes_filled_so_far += length;
          }

          bool isFull() {
            return bytes_filled_so_far == array.size();
          }
        };
        map<size_t, struct buffer> bufferMap;

      public:
        WriteSession(FileToken file_, size_t offset_, size_t bytes_)
          : file(CkpvAccess(manager)->get(file_))
          , token(file_)
          , sessionOffset(offset_)
          , myOffset((sessionOffset / file->opts.peStripe + thisIndex)
                     * file->opts.peStripe)
          , sessionBytes(bytes_)
          , myBytes(min(file->opts.peStripe, sessionOffset + sessionBytes - myOffset))
          , myBytesWritten(0)
        {
          CkAssert(file->fd != -1);
          CkAssert(myOffset >= sessionOffset);
          CkAssert(myOffset + myBytes <= sessionOffset + sessionBytes);
        }

        WriteSession(CkMigrateMessage *m) { }

        void forwardData(const char *data, size_t bytes, size_t offset) {
          CkAssert(offset >= myOffset);
          CkAssert(offset + bytes <= myOffset + myBytes);

          size_t stripeSize = file->opts.writeStripe;

          while (bytes > 0) {
            size_t stripeBase = (offset/stripeSize)*stripeSize;
            size_t stripeOffset = max(stripeBase, myOffset);
            size_t nextStripe = stripeBase + stripeSize;
            size_t expectedBufferSize = min(nextStripe, myOffset + myBytes) - stripeOffset;
            size_t bytesInCurrentStripe = min(nextStripe - offset, bytes);

            buffer& currentBuffer = bufferMap[stripeOffset];
            currentBuffer.expect(expectedBufferSize);

            currentBuffer.insertData(data, bytesInCurrentStripe, offset - stripeOffset);

            if (currentBuffer.isFull()) {
              flushBuffer(currentBuffer, stripeOffset);
              bufferMap.erase(stripeOffset);
            }

            bytes -= bytesInCurrentStripe;
            data += bytesInCurrentStripe;
            offset += bytesInCurrentStripe;
          }

          if (myBytesWritten == myBytes)
            contribute(CkCallback(CkIndex_WriteSession::syncData(), thisProxy));
        }

        void syncData() {
          int status;
          CkAssert(bufferMap.size() == 0);
#if CMK_HAS_FDATASYNC_FUNC
          while ((status = fdatasync(file->fd)) < 0) {
            if (errno != EINTR) {
              fatalError("fdatasync failed", file->name);
            }
          }
#elif CMK_HAS_FSYNC_FUNC
          while ((status = fsync(file->fd)) < 0) {
            if (errno != EINTR) {
              fatalError("fsync failed", file->name);
            }
          }
#elif defined(_WIN32)
          intptr_t hFile = _get_osfhandle(file->fd);
          if (FlushFileBuffers((HANDLE)hFile) == 0)
            fatalError("FlushFileBuffers failed", file->name);
#elif CMK_HAS_SYNC_FUNC
#warning "Will call sync() for every completed write"
          sync(); // No error reporting from sync()
#else
#warning "No file synchronization function available!"
#endif

          contribute(sizeof(FileToken), &token, CkReduction::max_int,
                     CkCallback(CkReductionTarget(Director, sessionComplete), director));
        }

        void flushBuffer(buffer& buf, size_t bufferOffset) {
          int l = buf.bytes_filled_so_far;
          char *d = &(buf.array[0]);

          CmiInt8 ret = CmiPwrite(file->fd, d, l, bufferOffset);
          if (ret < 0)
            fatalError("Call to pwrite failed", file->name);

          CkAssert(ret == l);
          myBytesWritten += l;
        }
      };

      class Map : public CBase_Map {
      public:
        Map()
          { }

        int procNum(int arrayHdl, const CkArrayIndex &element) {
          return 0;
        }
      };
    }

    void open(string name, CkCallback opened, Options opts) {
      impl::director.openFile(name, opened, opts);
    }

    void startSession(File file, size_t bytes, size_t offset,
                      CkCallback ready, CkCallback complete) {
      impl::director.prepareWriteSession(file.token, bytes, offset, ready, complete);
    }
    void startSession(File file, size_t bytes, size_t offset, CkCallback ready,
                      const char *commitData, size_t commitBytes, size_t commitOffset,
                      CkCallback complete) {
      impl::director.prepareWriteSession(file.token, bytes, offset, ready,
                                         commitData, commitBytes, commitOffset,
                                         complete);
    }

    void write(Session session, const char *data, size_t bytes, size_t offset) {
        using namespace impl;
        CkpvAccess(manager)->write(session, data, bytes, offset);
    }

    void close(File file, CkCallback closed) {
      impl::director.close(file.token, closed);
    }

    class SessionCommitMsg : public CMessage_SessionCommitMsg {

    };
  }
}

#include "CkIO.def.h"
#include "CkIO_impl.def.h"
