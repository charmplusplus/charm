#include <stdio.h>
#include "hello.decl.h"
#include "ckdll.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int nElements;

/*mainchare*/
class Main : public CBase_Main
{
public:
  Main(CkArgMsg* m)
  {
    //Process command-line arguments
    nElements=5;
    if(m->argc >1 ) nElements=atoi(m->argv[1]);
    delete m;

    //Start the computation
    CkPrintf("Running Hello on %d processors for %d elements\n",
	     CkNumPes(),nElements);
    mainProxy = thisProxy;

    CProxy_Hello arr = CProxy_Hello::ckNew(nElements);

    arr[0].SayHi(17);
  };

  void done(void)
  {
    CkPrintf("All done\n");
    CkExit();
  };
};

void foo(int i) {
  CkPrintf("Called foo(%d)\n",i);
}

/*array [1D]*/
class Hello : public CBase_Hello
{
public:
  Hello()
  {
    CkPrintf("Hello %d created\n",thisIndex);
  }

  Hello(CkMigrateMessage *m) {}
  
  void SayHi(int hiNo)
  {
    typedef char *(*sayHiFn)(int hiNo);
    char progBuf[1024]; /*Write a little program in this buffer*/
    strcpy(progBuf,
"#include <stdio.h>\n"
"void foo(int i);\n"
"extern \"C\"\n"
"char *sayHi_dll(int hiNo) {\n"
"   char *ret=new char[100];\n"
"   foo(hiNo);\n"
"   sprintf(ret,\"interpreted Hello[%d]\",hiNo);\n"
"   return ret;\n"
"}\n"
);

    {
      CkCppInterpreter interp(progBuf);
      sayHiFn sayHi_dll=(sayHiFn)interp.lookup("sayHi_dll");
      if (sayHi_dll==NULL) {
          CkError("CkCppInterpreter failed.\n");
          CkExit();
      }
      char *ret=sayHi_dll(hiNo);
      CkPrintf("%s from element %d\n",ret,thisIndex);
      delete[] ret;
    }
    if (thisIndex < nElements-1)
      //Pass the hello on:
      thisProxy[thisIndex+1].SayHi(hiNo+1);
    else 
      //We've been around once-- we're done.
      mainProxy.done();
  }
};

#include "hello.def.h"
