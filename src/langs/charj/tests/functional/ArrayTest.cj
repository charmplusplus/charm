
public mainchare ArrayTest {
  Array<int, 2> foobar = new Array<int, 2>([5, 3]);

  public entry ArrayTest(CkArgMsg[~]@ msg) {
    thisProxy@entry1();
  }

  public int id(int i) { return i; }

  public entry void entry1() {
    Array<int> foo = new Array<int>([10]);

    for (int i = 0; i < foo.size(); i++) {
      foo[i] = i;
    }

    for (int i = 0; i < foo.size()-1; i++) {
      foo[i+1] = i;
      foo[i/2 + (id(i) - i/2)] = i;
    }

    for (int i = 0; i < foo.size(); i++) {
      printf("foo[%d]: %d\n", i, foo[i]);
    }

    delete foo;

    int dim1 = 2;
    int dim2 = 3;
    Array<int, 2> bar = new Array<int, 2>([dim1, dim2]);

    for (int i = 0; i < dim1; i++) {
      for (int j = 0; j < dim2; j++) {
        bar[i, j] = i;
      }
    }

    for (int i = 0; i < dim1; i++) {
      for (int j = 0; j < dim2; j++) {
        printf("bar[%d, %d]: %d\n", i, j, bar[i, j]);
      }
    }

    delete bar;

    for (int i = 0; i < foobar.size(0); i++) {
      for (int j = 0; j < foobar.size(1); j++) {
        foobar[i, j] = 10 * i + j;
      }
    }

    for (int i = 0; i < foobar.size(0); i++) {
      for (int j = 0; j < foobar.size(1); j++) {
        printf("foobar[%d, %d]: %d\n", i, j, foobar[i, j]);
      }
    }

    delete foobar;

    Vector<int> baz = new Vector<int>(10);
    Matrix<double> I = new Matrix<double>(10);

    CkExit();
  }
}
