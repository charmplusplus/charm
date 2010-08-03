
public mainchare ArrayTest {
      Array<int> test1 = new Array<int>([10]);

      public entry ArrayTest(CkArgMsg msg) {
           thisProxy@entry1();
      }

      public entry void entry1() {
           Array<int> test2 = new Array<int>([10]);

	   for (int i = 0; i < test1.size(); i++) {
               test1[i] = i;
               test2[i] = i + 1;
           }

           for (int i = 0; i < test1.size(); i++) {
                printf("%d %d\n", test1[i], test2[i]);
           }

           delete test2;

           CkExit();
      }
}
