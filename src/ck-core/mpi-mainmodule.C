#include "mpi_main.decl.h"
#include "mpi-interoperate.h"

/*mainchare of mainmodule for interoperability*/
class mpi_main : public CBase_mpi_main
{
public:
  mpi_main(CkArgMsg* m) 
  {
    delete m;
    thisProxy.exit();
  };

  void exit() {
    CkExit();
  }
};
#include "mpi_main.def.h"

