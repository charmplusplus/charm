#include <stdio.h>

#include <vector>
#include <list>
#include <string>
#include "pup_stl.h"
typedef std::map<int,std::string> foo;
typedef std::pair<int,std::string> fooPair;

#include "hello.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int nElements;

/*mainchare*/
class Main : public CBase_Main
{
public:
  Main(CkArgMsg* m)
  {
	for (int i=0;i<m->argc;i++)
		CkPrintf("argv[%d]='%s'\n",i,m->argv[i]);
    //Process command-line arguments
    nElements=5;
    if(m->argc >1 ) nElements=atoi(m->argv[1]);
    delete m;

    //Start the computation
    CkPrintf("Running Hello on %d processors for %d elements\n",
	     CkNumPes(),nElements);
    mainProxy = thisProxy;

    CProxy_Hello arr = CProxy_Hello::ckNew(nElements);

    foo f;
    f.insert(fooPair(2,"first, 2"));
    f.insert(fooPair(3,"second, 3"));
    f.insert(fooPair(1,"third, 1"));
    f.insert(fooPair(5,"fourth, 5"));
    arr[0].SayHi(17,f);
  };

  void done(void)
  {
    CkPrintf("All done\n");
    CkExit();
  };
};

/*array [1D]*/
class Hello : public CBase_Hello 
{
public:
  Hello()
  {
    CkPrintf("Hello %d created\n",thisIndex);
  }

  Hello(CkMigrateMessage *m) {}
  
  void SayHi(int hiNo,const foo &f)
  {
    int i=0;
    for (foo::const_iterator it=f.begin();it!=f.end();++it)
      CkPrintf("f[%d]=(%d,'%s')\n",i++,it->first,it->second.c_str());
    CkPrintf("Hi[%d] from element %d\n",hiNo,thisIndex);
    if (thisIndex < nElements-1)
      //Pass the hello on:
      thisProxy[thisIndex+1].SayHi(hiNo+1,f);
    else 
      //We've been around once-- we're done.
      mainProxy.done();
  }
};

#include "hello.def.h"
