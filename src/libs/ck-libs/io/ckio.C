#include <string>
#include "CkIO.decl.h"
#include <ckio.h>
#include <errno.h>
#include <algorithm>
#include <sys/stat.h>
#include <fcntl.h>

#if defined(_WIN32)
#include <io.h>

int pwrite(int fd, const void *buf, size_t nbytes, off_t offset)
{
  long ret = _lseek(fd, offset, SEEK_SET);

  if (ret == -1) {
    return(-1);
  }
  return(_write(fd, buf, nbytes));
}
#define NO_UNISTD_NEEDED
#endif

#if defined(__PGIC__)
// PGI compilers define funny feature flags that lead to standard
// headers omitting this prototype
ssize_t pwrite(int fd, const void *buf, size_t count, off_t offset);
#define NO_UNISTD_NEEDED
#endif

#if !defined(NO_UNISTD_NEEDED)
#include <unistd.h>
#endif

namespace Ck { namespace IO {

    namespace impl {
      CProxy_Director director;
      Manager *manager;
    }

    class SessionReadyMsg : public CMessage_SessionReadyMsg {
      FileToken file;
      size_t bytes, offset;
      impl::CProxy_WriteSession proxy;
      friend class impl::Manager;
    public:
      SessionReadyMsg(FileToken file_, size_t bytes_, size_t offset_,
                      CkArrayID sessionID)
        : file(file_), bytes(bytes_), offset(offset_), proxy(sessionID)
      { }
    };

    namespace impl {
      using std::min;
      using std::max;
      using std::map;

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

      class Director : public CBase_Director {
        int filesOpened;
        map<FileToken, impl::FileInfo> files;
        CProxy_Manager managers;

      public:
        Director(CkArgMsg *m)
          : filesOpened(0)
        {
          delete m;
          director = thisProxy;
          managers = CProxy_Manager::ckNew();
        }

        void openFile(std::string name, CkCallback opened, Options opts) {
          if (-1 == opts.peStripe)
            opts.peStripe = 16 * 1024 * 1024;
          if (-1 == opts.writeStripe)
            opts.writeStripe = 4 * 1024 * 1024;
          if (-1 == opts.activePEs)
            opts.activePEs = min(CkNumPes(), 32);
          if (-1 == opts.basePE)
            opts.basePE = 0;
          if (-1 == opts.skipPEs)
            opts.skipPEs = CkMyNodeSize();

          files[filesOpened] = FileInfo(name, opts);
          managers.openFile(filesOpened++, name, opened, opts);
        }

        void prepareWriteSession(FileToken file, size_t bytes, size_t offset,
                                 CkCallback ready, CkCallback complete) {
          Options &opts = files[file].opts;

          // XXX: Replace with a direct calculation
          int numStripes = 0, o = offset;
          while (o < offset + bytes) {
            numStripes++;
            o += opts.peStripe - o % opts.peStripe;
          }

          CkArrayOptions sessionOpts(numStripes);
          sessionOpts.setMap(managers);
          CProxy_WriteSession session =
            CProxy_WriteSession::ckNew(file, bytes, offset, complete, sessionOpts);
          ready.send(new SessionReadyMsg(file, bytes, offset, session));
        }
      };

      class Manager : public CBase_Manager {

      public:
        Manager()
        {
          manager = this;
        }

        void openFile(FileToken token, std::string name,
                      CkCallback opened, Options opts) {
          CkAssert(files.end() == files.find(token));
          CkAssert(lastActivePE(opts) < CkNumPes());
          CkAssert(opts.writeStripe <= opts.peStripe);
          files[token] = impl::FileInfo(name, opts);

          // Open file if we're one of the active PEs
          // XXX: Or maybe wait until the first write-out, to smooth the metadata load?
          if (((CkMyPe() - opts.basePE) % opts.skipPEs == 0 &&
               CkMyPe() < lastActivePE(opts)) ||
              true) {
            files[token].fd = doOpenFile(name);
          }

          contribute(sizeof(FileToken), &token, CkReduction::max_int, opened);
        }

        void write(SessionReadyMsg *session,
                   const char *data, size_t bytes, size_t offset) {
          Options &opts = files[session->file].opts;
          size_t stripe = opts.peStripe;

          size_t sessionStripeBase = (session->offset / stripe) * stripe;

          while (bytes > 0) {
            size_t stripeIndex = (offset - sessionStripeBase) / stripe;
            size_t bytesToSend = min(bytes, stripe - offset % stripe);

            session->proxy[stripeIndex].forwardData(data, bytesToSend, offset);

            data += bytesToSend;
            offset += bytesToSend;
            bytes -= bytesToSend;
          }
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
        friend class WriteSession;

        int doOpenFile(const std::string& name) {
          int fd;
#if defined(_WIN32)
          fd = _open(name.c_str(), _O_WRONLY | _O_CREAT, _S_IREAD | _S_IWRITE);
#else
          fd = ::open(name.c_str(), O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR);
#endif
          if (-1 == fd)
            CkAbort("Failed to open a file for parallel output");
          return fd;
        }

        int lastActivePE(const Options &opts) {
          return opts.basePE + (opts.activePEs-1)*opts.skipPEs;
        }
      };

      class WriteSession : public CBase_WriteSession {
        const FileInfo *file;
        size_t sessionOffset, myOffset;
        size_t sessionBytes, myBytes, myBytesWritten;
        CkCallback complete;

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
        WriteSession(FileToken file_, size_t offset_, size_t bytes_, CkCallback complete_)
          : file(&manager->files[file_])
          , sessionOffset(offset_)
          , myOffset((sessionOffset / file->opts.peStripe + thisIndex)
                     * file->opts.peStripe)
          , sessionBytes(bytes_)
          , myBytes(min(file->opts.peStripe, sessionOffset + sessionBytes - myOffset))
          , myBytesWritten(0)
          , complete(complete_)
        { }

        WriteSession(CkMigrateMessage *m) { }

        void forwardData(const char *data, size_t bytes, size_t offset) {
          //files[token].bufferMap[(offset/stripeSize)*stripeSize] is the buffer to which this char should write to.

          CkAssert(offset >= myOffset);
          CkAssert(offset + bytes < myOffset + myBytes);

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
          fdatasync(file->fd);
          contribute(complete);
        }

        void flushBuffer(buffer& buf, size_t bufferOffset) {
          int l = buf.bytes_filled_so_far;
          char *d = &(buf.array[0]);

          while (l > 0) {
            CmiInt8 ret = pwrite(file->fd, d, l, bufferOffset);
            if (ret < 0) {
              if (errno == EINTR) {
                continue;
              } else {
                CkPrintf("Output failed on PE %d: %s", CkMyPe(), strerror(errno));
                CkAbort("Giving up");
              }
            }
            l -= ret;
            d += ret;
            bufferOffset += ret;
          }
          myBytesWritten += buf.bytes_filled_so_far;
        }
      };
    }

    void open(std::string name, CkCallback opened, Options opts) {
      impl::director.openFile(name, opened, opts);
    }

    void startSession(FileToken token, size_t bytes, size_t offset,
                      CkCallback ready, CkCallback complete) {
      impl::director.prepareWriteSession(token, bytes, offset, ready, complete);
    }

    void write(SessionReadyMsg *session,
               const char *data, size_t bytes, size_t offset) {
      impl::manager->write(session, data, bytes, offset);
    }

    class SessionCommitMsg : public CMessage_SessionCommitMsg {

    };
  }
}

#include "CkIO.def.h"
