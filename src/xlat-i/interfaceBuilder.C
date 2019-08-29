
#include "interfaceBuilder.h"

/* This is an example of how to use the XI-builder library
 * and should not be compiled or linked to otherwise.
 *
 * The three example programs contained here are:
 *   Test, Hello1D, and FibSDAG
 */

namespace Builder {
  xi::AstChildren<xi::Module>* generateTestAst() {
    File* f = new File();
    Module* m = new Module("test", true);

    MainChare* mc = new MainChare("Main");
    ConsEntry* mccons = new ConsEntry("Main");
    mccons->addEntryParameter(new Parameter(new PtrType(new Type("CkArgMsg")), "m"));
    mc->addEntry(mccons);

    Chare* c = new Chare("Fib");
    ConsEntry* cons = new ConsEntry("Fib");
    c->addEntry(cons);
    Entry* entry1 = new Entry(new Type("void"), "sendValue");
    entry1->addEntryParameter(new Parameter(new Type("int"), "value"));
    c->addEntry(entry1);

    Readonly* ro = new Readonly(new Type("int"), "abc");

    m->addModuleEntity(mc);
    m->addModuleEntity(c);
    m->addModuleEntity(ro);

    f->addModule(m);
    return f->generateAst();
  }

  xi::AstChildren<xi::Module>* generateHello1d() {
    File* f = new File();
    Module* m = new Module("hello", true);

    MainChare* mc = new MainChare("Main");
    ConsEntry* mccons = new ConsEntry("Main");
    mccons->addEntryParameter(new Parameter(new PtrType(new Type("CkArgMsg")), "m"));
    mc->addEntry(mccons);
    Entry* entryDone = new Entry(new Type("void"), "done");
    mc->addEntry(entryDone);

    Array* a = new Array("Hello", "1D");
    ConsEntry* cons = new ConsEntry("Hello");
    a->addEntry(cons);
    Entry* entrySayHi = new Entry(new Type("void"), "SayHi");
    entrySayHi->addEntryParameter(new Parameter(new Type("int"), "hiNo"));
    a->addEntry(entrySayHi);

    Readonly* ro1 = new Readonly(new Type("int"), "nElements");
    Readonly* ro2 = new Readonly(new Type("CProxy_Main"), "mainProxy");

    m->addModuleEntity(mc);
    m->addModuleEntity(a);
    m->addModuleEntity(ro1);
    m->addModuleEntity(ro2);

    f->addModule(m);
    return f->generateAst();
  }

  xi::AstChildren<xi::Module>* generateFibSDAG() {
    File* f = new File();
    Module* m = new Module("fib", true);

    MainChare* mc = new MainChare("Main");
    ConsEntry* mccons = new ConsEntry("Main");
    mccons->addEntryParameter(new Parameter(new PtrType(new Type("CkArgMsg")), "m"));
    mc->addEntry(mccons);

    Chare* a = new Chare("Fib");
    ConsEntry* cons = new ConsEntry("Fib");
    cons->addEntryParameter(new Parameter(new Type("int"), "n"));
    cons->addEntryParameter(new Parameter(new Type("bool"), "isRoot"));
    cons->addEntryParameter(new Parameter(new Type("CProxy_Fib"), "parent"));
    a->addEntry(cons);

    Entry* entryResponse = new Entry(new Type("void"), "response");
    entryResponse->addEntryParameter(new Parameter(new Type("int"), "val"));
    a->addEntry(entryResponse);

    Entry* entryCalc = new Entry(new Type("void"), "calc");
    entryCalc->addEntryParameter(new Parameter(new Type("int"), "n"));

    SDAG::Sequence* calcSDAG = new SDAG::Sequence();
    SDAG::Serial* constructChildren = new
      SDAG::Serial("CProxy_Fib::ckNew(n - 1, false, thisProxy);"
                   "CProxy_Fib::ckNew(n - 2, false, thisProxy);");
    SDAG::Serial* respond = new SDAG::Serial("respond(val + val2);");

    SDAG::SEntry* response1 = new SDAG::SEntry("response");
    response1->addEntryParameter(new Parameter(new Type("int"), "val"));
    SDAG::SEntry* response2 = new SDAG::SEntry("response");
    response2->addEntryParameter(new Parameter(new Type("int"), "val2"));

    SDAG::When* whenResponse2 = new SDAG::When(new SDAG::Sequence(respond));
    whenResponse2->addSEntry(response2);
    SDAG::When* whenResponse1 = new SDAG::When(new SDAG::Sequence(whenResponse2));
    whenResponse1->addSEntry(response1);

    SDAG::Sequence* elseSeq = new SDAG::Sequence();
    elseSeq->addConstruct(constructChildren);
    elseSeq->addConstruct(whenResponse1);
    SDAG::Else* elseGTThresh = new SDAG::Else(elseSeq);

    SDAG::Serial* respondSeq = new SDAG::Serial("respond(seqFib(n));");
    SDAG::If* ifThesh = new SDAG::If("n < THRESHOLD", respondSeq, elseGTThresh);

    calcSDAG->addConstruct(ifThesh);
    entryCalc->addSDAG(calcSDAG);

    a->addEntry(entryCalc);

    m->addModuleEntity(mc);
    m->addModuleEntity(a);

    f->addModule(m);
    return f->generateAst();
  }
}

#include "xi-main.h"

int main(int argc, char* argv[]) {
  //xi::AstChildren<xi::Module>* ast = Builder::generateHello1d();
  xi::AstChildren<xi::Module>* ast = Builder::generateFibSDAG();
  processAst(ast, false, false, 0, 0, "generate", "");
}
