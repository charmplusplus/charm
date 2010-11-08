#include "Array.h"
#include <iostream>
#include <list>
#include <string>

using namespace std;
using namespace CharjArray;

class ArrayTest {
public:
  virtual string testName() = 0;
  virtual bool test() = 0;
};

class Array1DUnit : public ArrayTest {
public:
  string testName() {
    return string("1D int array test");
  }

  bool test() {
    int size = 10;
    Range r(size);
    Array<int> *array1 = new Array<int>(Domain<1>(Range(size)));
    Array<int> &ref = *array1;

    if (ref.size() != size)
      return false;

    for (int i = 0; i < ref.size(); i++) {
      ref[i] = 20 + i;
    }

    for (int i = 0; i < size; i++) {
      if (ref[i] != i + 20)
        return false;
      //cout << i << ": " << ref[i] << endl;
    }

    return true;
  }
};

class Array2DUnitLinear : public ArrayTest {
public:
  string testName() {
    return string("2D int array test");
  }

  bool test() {
    int size1 = 2;
    int size2 = 3;
    Range dim1(size1);
    Range dim2(size2);
    Range rarray [2] = {dim1, dim2};
    Array<int, 2> *array1 = new Array<int, 2>(Domain<2>(rarray));
    Array<int, 2> &ref = *array1;

    if (ref.size() != size1 * size2)
      return false;

    for (int i = 0; i < ref.size(); i++) {
      ref[i] = 20 + i;
    }

    for (int i = 0; i < size1 * size2; i++) {
      if (ref[i] != i + 20)
        return false;
      //cout << i << ": " << ref[i] << endl;
    }

    return true;
  }
};

class Array2DUnitRowMajor : public ArrayTest {
public:
  string testName() {
    return string("2D int array test");
  }

  bool test() {
    int size1 = 2;
    int size2 = 3;
    Range dim1(size1);
    Range dim2(size2);
    Range rarray [2] = {dim1, dim2};
    Array<int, 2> *array1 = new Array<int, 2>(Domain<2>(rarray));
    Array<int, 2> &ref = *array1;

    if (ref.size() != size1 * size2)
      return false;

    for (int i = 0; i < ref.size(0); i++) {
      for (int j = 0; j < ref.size(1); j++) {
        ref.access(i, j) = i;
      }
    }

    for (int i = 0; i < size2; i++) {
      if (ref[i] != 0)
        return false;
    }

    for (int i = size2; i < size2 * 2; i++) {
      if (ref[i] != 1)
        return false;
    }

    return true;
  }
};

static void print_result(ArrayTest *test)
{
  bool status = test->test();
  cout << test->testName() << (status ? " PASSED" : " FAILED") << endl;
}

int main(void) {
  list<ArrayTest*> tests;
  Array1DUnit a1;
  Array2DUnitLinear a2;
  Array2DUnitRowMajor a3;

  tests.push_back(&a1);
  tests.push_back(&a2);
  tests.push_back(&a3);

  for_each(tests.begin(), tests.end(), print_result);

#if 0
  Array<int>* newTest = new Array<int>(Domain<1>(Range(10)));

  for (int i = 0; i < 9; i++) {
    (*newTest)[i] = i;
  }

  for (int i = 0; i < 10; i++) {
    cout << "newTest: " << (*newTest)[i] << endl;
  }

  Array<int>* newTest2 = (*newTest)[Domain<1>(Range(5, 9))];

  for (int i = 0; i < newTest2->size(); i++) {
    cout << "newTest2: " << (*newTest2)[i] << endl;
  }
#endif

  return 0;
}
