
public mainchare ArrayTest {
      public entry ArrayTest(CkArgMsg msg) {
           thisProxy@entry1();
      }

      public entry void entry1() {
           Array<int> test = new Array<int>([10]);

	   for (int i = 0; i < test.size(); i++) {
               test[i] = i;
           }

           for (int i = 0; i < test.size(); i++) {
                printf("%d\n", test[i]);
           }

           delete test;

           CkExit();
      }
}
