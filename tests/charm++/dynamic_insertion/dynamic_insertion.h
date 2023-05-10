#include "dynamic_insertion.decl.h"

class Main : public CBase_Main {
 private:
  int iteration{0};
  int sum_of_indices{0}; 
 public:
  Main(CkArgMsg* msg);
  void initialize();
  void create_new_elements();
  void ping_elements();
  void exit();
  void check_sum(int sum);
  void begin_array_insertion();
  void end_array_insertion();
};

class DgGroup : public CBase_DgGroup {
public:
  DgGroup();
  void begin_array_insertion();
  void end_array_insertion();
  void create_new_element(int iteration, int new_id);
};

class DgElement : public CBase_DgElement {
 public:
  DgElement(int iteration);
  void ping(int iteration);
  void create_new_element(int iteration, int new_id);
private:
  int iteration_at_creation;
};
