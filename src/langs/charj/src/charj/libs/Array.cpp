#include "Array.h"
#include <iostream>

using namespace std;
using namespace CharjArray;

class Test {
public: Array<int> aaa;
  
  Test() {
    aaa.init(Domain<1>(Range(3)));
  }

};

class ArrayTest2 {
public: 
  Array<int> test;

  void test1() {
    test[0] = 0;
    int i;
    for (i = 0; i < 10; i++) {
      test[i] = 110;
    }
  }

  ArrayTest2() {
    test.init(Domain<1>(Range(10)));
  }
};

int main(void) {
  Range ranges[1];
  ranges[0] = 10;

  Domain<1> domain(ranges);

  Array<int> arr;

  arr.init(domain);

  //arr[0] = 10;
  arr[Point<1>(0)] = 10000;
  arr[Point<1>(1)] = 20;
  arr[Point<1>(8)] = 200;

  //Array<int> aaa;
  //aaa(Domain<1>(Range(3));

  for (int i = 0; i < arr.size(); i++) {
    cout << i << " -> " << arr[i] << endl;
  }

  cout << endl;

  ArrayTest2 at;
  at.test1();

  for (int i = 0; i < at.test.size(); i++) {
    cout << i << " -> " << at.test[i] << endl;
  }

  return 0;
}
