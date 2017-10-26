#define _GNU_SOURCE
#include <sys/uio.h>

// Method checks if process has permissions to use CMA
void CmiInitCma(void) {
  char buffer   = '0';
  int cma_works = 0;
  int fd;

  // determine permissions
  fd = open("/proc/sys/kernel/yama/ptrace_scope", O_RDONLY);
  if (0 <= fd) {
    read (fd, &buffer, 1);
    close(fd);
  }

  if('0' != buffer) {
#if defined PR_SET_PTRACER
    // set ptrace scope to allow attach
    int ret = prctl (PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);
    if (0 == ret) {
      cma_works = 1;
    }
#endif
  } else {
    cma_works = 1;
  }

  if(cma_works) {
    if(CmiMyPe() == 0) {
      CmiPrintf("Charm++> cma enabled for within node transfers using the zerocopy API\n");
    }
  } else {
    CmiAbort("Can't use CMA for SHM! Permissions Denied\n");
  }
}

/* Used to read from a list of remote buffers and write into a
 * list of local buffers over SHM using the cma library
 */
int readShmCma(
  pid_t remote_pid,
  struct iovec *local,
  struct iovec *remote,
  int numOps,
  size_t total_bytes) {

  int nread = process_vm_readv(remote_pid, local, numOps, remote, numOps, 0);
  if(nread != total_bytes) {
    CmiAbort("process_vm_readv failed!\n");
    return errno;
  }
  return 0;
}

/* Used to write into a list of remote buffers by reading from a
 * list of local buffers over SHM using the cma library
 */
int writeShmCma(
  pid_t remote_pid,
  struct iovec *local,
  struct iovec *remote,
  int numOps,
  size_t total_bytes) {

  int nread = process_vm_writev(remote_pid, local, numOps, remote, numOps, 0);
  if(nread != total_bytes) {
    CmiAbort("process_vm_writev failed!\n");
    return errno;
  }
  return 0;
}

