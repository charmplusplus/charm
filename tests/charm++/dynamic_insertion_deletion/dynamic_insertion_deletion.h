#include <cstddef>
#include <unordered_map>

using ElementId_t = int;

#include "dynamic_insertion_deletion.decl.h"

class Main : public CBase_Main {
 private:
  size_t iteration{0};
  std::unordered_map<ElementId_t, int> proc_map{};
  std::unordered_map<ElementId_t, int> initial_proc_map{};
  std::unordered_map<ElementId_t, int> possible_hangs{};
 public:
  Main(CkArgMsg* msg);
  void initialize();
  void check_domain();
  void create_new_elements();
  void delete_old_elements();
  void exit();
  void check_volume(const double volume);
  void build_proc_map();
  void add_proc_to_map(const ElementId_t& id, const int proc);
  void remove_proc_from_map(const ElementId_t& id, const int proc);
  void ping_elements();
};

class DgElement : public CBase_DgElement {
 public:
  DgElement(const size_t iteration);
  void create_new_elements(const size_t iteration);
  void delete_old_elements(const size_t iteration);
  void send_volume(const size_t iteration);
  void receive_ping(const size_t iteration);
  void send_proc_to_main(const size_t iteration);

 private:
  size_t iteration_at_creation;
};
