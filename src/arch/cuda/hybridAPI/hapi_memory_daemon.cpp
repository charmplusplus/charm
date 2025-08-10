#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <algorithm>
#include <queue>
#include <atomic>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sched.h>

#include <cuda_runtime.h>
#include <cuda.h>

#define CUDA_CHECK(call) do { \
  cudaError_t err = call; \
  if (err != cudaSuccess) { \
    fprintf(stderr, "HAPI> CUDA call failed: %s\n", cudaGetErrorString(err)); \
  } \
} while(0)


#define SERVER_FIFO_TEMPLATE "/tmp/server_pipe_%ld"
#define CLIENT_FIFO_TEMPLATE "/tmp/client_pipe_%ld"
#define BUFFER_SIZE 256
#define STREAM_BUF_SIZE 1024

// Managing memory state in server
std::unordered_map<int, std::pair<void*, size_t>> hapiMemoryMap;
int allocId = 0;

void hapiProcessMemoryRequest(int server_fd, int my_device, char* buf)
{
  long client_pid;
  char command[BUFFER_SIZE];
  sscanf(buf, "%[^:]:", command);

  char* pid_str = strchr(buf, ':');
  if (pid_str) client_pid = atol(pid_str + 1); else return;

  printf("HAPI> Processing memory request: %s from client %ld\n", command, client_pid);

  char client_fifo_path[BUFFER_SIZE];
  sprintf(client_fifo_path, CLIENT_FIFO_TEMPLATE, client_pid);
  int client_fd = open(client_fifo_path, O_WRONLY);

  if (strcmp(command, "CKPT") == 0) 
  {
    int client_pe, size;
    sscanf(buf, "CKPT:%ld:%d:%d:", &client_pid, &client_pe, &size);

    printf("HAPI> Checkpoint request from client %ld (PE %d): %d bytes\n", client_pid, client_pe, size);
    fflush(stdout);

    char* handle_start = strchr(buf, ':');
    handle_start = strchr(handle_start + 1, ':');
    handle_start = strchr(handle_start + 1, ':') + 1;

    cudaIpcMemHandle_t ipc_handle;
    memcpy(&ipc_handle, handle_start, sizeof(cudaIpcMemHandle_t));

    void* client_ptr;
    CUDA_CHECK(cudaIpcOpenMemHandle(&client_ptr, ipc_handle, cudaIpcMemLazyEnablePeerAccess));

    auto allocation = std::make_pair(nullptr, size);
    CUDA_CHECK(cudaMalloc((void**) &(allocation.first), size));

    CUDA_CHECK(cudaMemcpy((void*) allocation.first, client_ptr, size, cudaMemcpyDeviceToDevice));
    hapiMemoryMap[allocId] = allocation;

    CUDA_CHECK(cudaIpcCloseMemHandle(client_ptr));
    write(client_fd, &allocId, sizeof(int));
    allocId++;
  }
  else if (strcmp(command, "GET") == 0)
  {
    int alloc_id;
    sscanf(buf, "GET:%ld:%d", &client_pid, &alloc_id);

    void* ptr = hapiMemoryMap[alloc_id].first;
    cudaIpcMemHandle_t ipc_handle;
    CUDA_CHECK(cudaIpcGetMemHandle(&ipc_handle, ptr));
    write(client_fd, &ipc_handle, sizeof(cudaIpcMemHandle_t));
  }
  else if (strcmp(command, "FREE") == 0)
  {
    int alloc_id;
    sscanf(buf, "FREE:%ld:%d", &client_pid, &alloc_id);

    auto it = hapiMemoryMap.find(alloc_id);
    if (it != hapiMemoryMap.end()) {
      CUDA_CHECK(cudaFree(it->second.first));
      hapiMemoryMap.erase(it);
    }
  }
  else if (strcmp(command, "KILL") == 0)
  {
    printf("Server: KILL command received from client %ld\n", client_pid);
    write(client_fd, "\0", 1);
    close(server_fd);

    char server_fifo[BUFFER_SIZE];
    sprintf(server_fifo, SERVER_FIFO_TEMPLATE, my_device);
    if (remove(server_fifo) == 0) {
        printf("File '%s' deleted successfully.\n", server_fifo);
    } else {
        printf("Error deleting file '%s': %s\n", server_fifo, strerror(errno));
    }
    exit(0);
  }

  close(client_fd);
}

void hapiStartMemoryDaemon(int my_device) {

  int current_cpu = sched_getcpu();
  printf("Daemon: Current CPU is %d\n", current_cpu);

  // Child process (daemon)
  printf("DAEMON: Starting daemon process PID=%d\n", getpid());
  
  // Set up the daemon's CUDA context
  cudaSetDevice(my_device);

  char server_fifo[BUFFER_SIZE];
  sprintf(server_fifo, SERVER_FIFO_TEMPLATE, my_device);
  mkfifo(server_fifo, 0666);
  
  // Open server FIFO for reading (this may block until a writer connects)
  char server_fifo_path[BUFFER_SIZE];
  sprintf(server_fifo_path, SERVER_FIFO_TEMPLATE, my_device);
  printf("DAEMON: Opening server FIFO %s\n", server_fifo_path);
  int server_fd = open(server_fifo_path, O_RDONLY | O_NONBLOCK);
  if (server_fd == -1) {
    perror("DAEMON: open server FIFO");
    exit(1);
  }
  
  // Make it blocking for actual reads
  int flags = fcntl(server_fd, F_GETFL);
  fcntl(server_fd, F_SETFL, flags & ~O_NONBLOCK);

  char ready_fifo_path[BUFFER_SIZE];
  sprintf(ready_fifo_path, "/tmp/daemon_ready_%d", my_device);
  
  // Signal parent that daemon is ready
  int ready_fd = open(ready_fifo_path, O_WRONLY);
  if (ready_fd == -1) {
    perror("DAEMON: open ready FIFO for writing");
    exit(1);
  }
  write(ready_fd, "1", 1);
  close(ready_fd);
  
  printf("DAEMON: Ready signal sent to parent\n");
  
  // Main daemon loop
  char stream_buf[STREAM_BUF_SIZE];
  size_t data_in_stream = 0;
  int bytes_read;

  while (1)
  {
    // read() will block here until data is available
    bytes_read = read(server_fd, stream_buf + data_in_stream, 
                              STREAM_BUF_SIZE - data_in_stream);

    if (bytes_read > 0)
    {
      printf("DAEMON: Read %d bytes from server FIFO\n", bytes_read);
      data_in_stream += bytes_read;
      
      if (data_in_stream >= STREAM_BUF_SIZE) {
        printf("DAEMON: Stream buffer overflow");
        exit(1);
      }

      // Process all complete messages in the buffer
      while (1)
      {
        char* msg_end = (char*)memchr(stream_buf, '\0', data_in_stream);
        if (msg_end == NULL) {
          break; // Wait for more data
        }

        size_t msg_len = (msg_end - stream_buf) + 1;
        char current_request[BUFFER_SIZE];
        memcpy(current_request, stream_buf, msg_len);
        
        // Process the request. Note: This may exit on a KILL command.
        hapiProcessMemoryRequest(server_fd, my_device, current_request);

        // Remove processed message from buffer
        data_in_stream -= msg_len;
        memmove(stream_buf, stream_buf + msg_len, data_in_stream);
      }
    }
    else if (bytes_read == 0)
    {
      // A writer closed the connection. The FIFO is still open.
      // The next read() will block until a new writer connects.
      // A small sleep prevents a potential tight spin-loop on misconfiguration.
      usleep(1000);
    }
    else // bytes_read < 0
    {
      // An error occurred.
      if (errno == EINTR) {
        continue; // Interrupted by a signal, just try again.
      }
      perror("DAEMON: read from server FIFO");
      break; // Exit on fatal error.
    }
  }
  
  close(server_fd);
  exit(0);
}

int main(int argc, char** argv) {
  const char* local_rank_str = getenv("SLURM_LOCALID");
  int local_rank = atoi(local_rank_str);
  hapiStartMemoryDaemon(local_rank);
}