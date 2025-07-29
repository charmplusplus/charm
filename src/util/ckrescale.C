int arg_shrinkexpand = 0; // Global variable to control shrink-expand behavior
bool shrinkexpand_exit = false; // Flag to indicate if we are in the process of shrinking/expanding
bool in_restart = false; // Flag to indicate if we are in a restart process

void set_arg_shrinkexpand(int value) {
  arg_shrinkexpand = value;
}

int get_arg_shrinkexpand() {
  return arg_shrinkexpand;
}

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