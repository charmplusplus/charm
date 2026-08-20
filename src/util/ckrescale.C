#include "ckrescale.h"
#include "conv-ccs.h"

// The CCS request that asked for the rescale, held open until the rescale
// completes so the caller can be told it finished. It lives here rather than
// in a machine layer because the Reconverse build has none, and every build
// links ckrescale.
CcsDelayedReply shrinkExpandreplyToken;

// This rank's id in the membership being committed, computed during the load
// balancing step that precedes a rescale. Kept alongside the reply token for
// the same reason: the Reconverse build has no machine layer to hold it.
int mynewpe = 0;

bool shrinkexpand_exit = false; // Flag to indicate if we are in the process of shrinking/expanding
bool in_restart = false; // Flag to indicate if we are in a restart process


void set_shrinkexpand_exit(bool value) {
  shrinkexpand_exit = value;
}

bool get_shrinkexpand_exit() {
  return shrinkexpand_exit;
}

void set_in_restart(bool value) {
  in_restart = value;
}

bool get_in_restart() {
  return in_restart;
}