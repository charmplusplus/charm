////////////////////////////////////
//
//  cksequence_test.C
//
//  Definition of chares in cksequence_test
//
//
////////////////////////////////////

#include "cksequence_test.decl.h"

#include "cksequence.h"
#include <vector>
#include  <set>

void FillRandomSeq(std::set<int> &vec) {
  srand(1024);
  int total_elements = 1000000;
  int max_element = 10000;
  int counter = total_elements;

  while (counter-- > 0) {
    vec.insert(rand() % max_element);
  }
}


class Main: public CBase_Main {
  public:
    Main(CkArgMsg *m) {
      std::set<int> vec;
      FillRandomSeq(vec);
      CkSequence<int> seq(vec.begin(), vec.end());
      seq.DoneInserting();

      CkSequence<int>::iterator it;
      CkSequence<int>::iterator it_end = seq.end();

      std::set<int>::iterator set_it = vec.begin();
      std::set<int>::iterator set_it_end = vec.end();

      for (it = seq.begin(); it != seq.end(); ++it, ++set_it) {
        if (*it != *set_it) {
          CkPrintf("Irregularities %d : %d\n", *it, *set_it);
        }
      }
      CkPrintf("Test passed!!\n");

      delete m;
      CkExit();
    };

};

#include "cksequence_test.def.h"
