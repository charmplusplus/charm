#include "converse.h"

struct Arg {
  int size;
  char data[4];
};

typedef CmiHandler lrpc_handler;

void lrpc_init(void);
int register_lrpc(lrpc_handler func);
void lrpc(int node, int funcnum, int prio, int stksiz, void *in, void *out);
void quick_lrpc(int node, int funcnum, void *in, void *out);
void async_lrpc(int node, int funcnum, void *in);
