#include "completion_test.decl.h"
#include "completion.decl.h"
#include "completion.h"
#include "megatest.h"

void completion_test_init(void){
  CProxy_completion_tester::ckNew();
}

void completion_test_moduleinit(void){} 


struct completion_tester : public CBase_completion_tester {
    CProxy_CompletionDetector detector;
    completion_tester() { thisProxy.run_test(); }

    completion_tester_SDAG_CODE
};

struct completion_array : public CBase_completion_array {
    completion_array(CProxy_CompletionDetector det, int n) {
	det.ckLocalBranch()->produce(thisIndex+1);
	det.ckLocalBranch()->done();
	det.ckLocalBranch()->consume(n - thisIndex);
    }
    completion_array(CkMigrateMessage *) {}
};

MEGATEST_REGISTER_TEST(completion_test,"phil",0)

#include "completion_test.def.h"
