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