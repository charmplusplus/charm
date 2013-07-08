#include <ckio.h>
#include <errno.h>
#include <algorithm>
#include <sys/stat.h>

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
    
    void write(SessionToken token, const char *data, size_t bytes, size_t offset) {
      Options &opts = files[token].opts;
      while (bytes > 0) {
        size_t stripeIndex = offset / opts.peStripe;
        int peIndex = stripeIndex % opts.activePEs;
        int pe = opts.basePE + peIndex * opts.skipPEs;
        size_t bytesToSend = std::min(bytes, opts.peStripe - offset % opts.peStripe);
	token.proxy[pe].forwardData(token, data, bytesToSend, offset);
	data += bytesToSend;
	offset += bytesToSend;
	bytes -= bytesToSend;
      }
    }

    namespace impl {
      CProxy_Director director;

      class Director : public CBase_Director {
        int filesOpened;
        std::map<FileToken, impl::FileInfo> files;

      public:
        Director()
          : filesOpened(0)
        {
          director = thisProxy;
        }

        void openFile(std::string name, CkCallback opened, Options opts) {
          if (-1 == opts.peStripe)
            opts.peStripe = 16 * 1024 * 1024;
          if (-1 == opts.writeStripe)
            opts.writeStripe = 4 * 1024 * 1024;

          if (-1 == opts.activePEs) {
            size_t numStripes = (bytes + opts.peStripe - 1) / opts.peStripe;
            opts.activePEs = std::min((size_t)CkNumNodes(), numStripes);
          }
          if (-1 == opts.basePE)
            opts.basePE = 0;
          if (-1 == opts.skipPEs)
            opts.skipPEs = CkMyNodeSize();

          CkAssert(lastActivePE(opts) < CkNumPes());
          CkAssert(opts.writeStripe <= opts.peStripe);

          files[filesOpened] = impl::FileInfo(name, opts);
          managers.openFile(filesOpened++, name, opened, opts);
        }

        void prepareWriteSession(FileToken file, size_t bytes, size_t offset,
                                 CkCallback ready, CkCallback complete) {
          int numElements = files[file].activePEs;
          CkArrayOpts opts(numElements);
          opts.setMap();
          CProxy_WriteSession session =
            CProxy_WriteSession::ckNew(file, bytes, offset, complete, opts);
          ready.send(new SessionReadyMessage(session));
        }
      };

      Manager *manager;

      class Manager : public CBase_Manager {

      public:
        Manager()
        {
          manager = this;
        }

        void openFile(FileToken token, std::string name,
                      CkCallback opened, Options opts) {
          CkAssert(files.end() == files.find(token));
          files[token] = impl::FileInfo(name, opts);

          // Open file if we're one of the active PEs
          // XXX: Or maybe wait until the first write-out, to smooth the metadata load?
          if (((CkMyPe() - opts.basePE) % opts.skipPEs == 0 &&
               CkMyPe() < lastActivePE(opts)) ||
              true) {
            files[token].fd = doOpenFile(name);
          }

          contribute(sizeof(FileToken), token, CkReduction::max_int, opened);
        }

      private:
        std::map<FileToken, impl::FileInfo> files;

        int doOpenFile(const std::string& name) {
          int fd;
#if defined(_WIN32)
          fd = _open(name.c_str(), _O_WRONLY | _O_CREAT, _S_IREAD | _S_IWRITE);
#else
          fd = open(name.c_str(), O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR);
#endif
          if (-1 == fd)
            CkAbort("Failed to open a file for parallel output");
          return fd;
        }
      };

      class WriteSession : public CBase_WriteSession {
        FileToken file;
        size_t sessionOffset;
        size_t sessionBytes, myBytes, myBytesWritten;
        CkCallback complete;

      public:
        WriteSession(FileToken file_, size_t offset_, size_t bytes_, CkCallback complete_)
          : file(file_), offset(offset_), bytes(bytes_), complete(complete_)
        { }

        void forwardData(SessionToken token, const char *data, size_t bytes,
                         size_t offset) {
          //files[token].bufferMap[(offset/stripeSize)*stripeSize] is the buffer to which this char should write to.
          CkAssert(offset + bytes <= files[token].bytes);
          // XXX: CkAssert(this is the right processor to receive this data)

          size_t stripeSize = files[token].opts.peStripe;   
          while(bytes > 0) {
            size_t stripeOffset = (offset/stripeSize)*stripeSize;
            size_t expectedBufferSize = std::min(files[token].bytes - stripeOffset, stripeSize);
            struct impl::buffer & currentBuffer = files[token].bufferMap[stripeOffset];
            size_t bytesInCurrentStripe = std::min(expectedBufferSize - offset%stripeSize, bytes);

            //check if buffer this element already exists in map. If not, insert and resize buffer to stripe size
            currentBuffer.expect(expectedBufferSize);

            currentBuffer.insertData(data, bytesInCurrentStripe, offset % stripeSize);

            // Ready to flush?
            if(currentBuffer.isFull()) {
              //initializa params
              int l = currentBuffer.bytes_filled_so_far;
              char *d = &(currentBuffer.array[0]);
              size_t bufferOffset = stripeOffset;
              //write to file loop
              while (l > 0) {
                CmiInt8 ret = pwrite(files[token].fd, d, l, bufferOffset);
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
              //write complete - remove this element from bufferMap and call dataWritten
              thisProxy[0].write_dataWritten(token, currentBuffer.bytes_filled_so_far);
              files[token].bufferMap.erase(stripeOffset);
            }

            bytes -= bytesInCurrentStripe;
            data += bytesInCurrentStripe;
            offset += bytesInCurrentStripe;
          }
        }
      };
    }
  }
}

#include "CkIO.def.h"
