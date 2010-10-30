#include "inlineem.decl.h"
#include "megatest.h"

readonly<CProxy_inline_tester> t;

void inlineem_moduleinit(void)
{
  CProxy_inline_group g = CProxy_inline_group::ckNew();
  t = CProxy_inline_tester::ckNew(g);
}

void inlineem_init(void)
{
  ((CProxy_inline_tester)t).start_test();
}

struct inline_group : public CBase_inline_group
{
  bool called;
  inline_group() : called(false) {}
  void try_inline() {
    called = true;
  }
};

struct inline_tester : public CBase_inline_tester
{
  CProxy_inline_group g;
  inline_tester(CProxy_inline_group g_) : g(g_) {}
  void start_test()
  {
    inline_group *gl = g.ckLocalBranch();
    if (gl->called)
      CkAbort("Inline test was called already?\n");

    g[CkMyPe()].try_inline();

    if (!gl->called)
      CkAbort("Inline test should have been called by now!\n");

    gl->called = false;
    megatest_finish();
  }
};

MEGATEST_REGISTER_TEST(inlineem, "phil", 0)

#include "inlineem.def.h"
