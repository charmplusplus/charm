#include "templates.h"
#include "templates.def.h"

readonly<CkGroupID> templates_redid;
CProxy_templates_Array<int> templatesArray;

int BINSIZE;

void templates_init(void) 
{
  int i;
  for(i=0;i<CkNumPes();i++) {
    CProxy_templates_Collector<int>::ckNew(i);
  }
  templatesArray[1].remoteRecv(new templates_Single<int>(7));
  int arr[3]; arr[0]=123; arr[1]=234; arr[2]=345;
  templatesArray[0].marshalled(3,arr);
}

void templates_moduleinit(void)
{
  templates_redid = CProxy_templates_Reduction<int>::ckNew();
  templatesArray = CProxy_templates_Array<int>::ckNew(2);
}

template <class dtype> void 
templates_Reduction<dtype>::submit(templates_Single<dtype> *msg)
{
  CProxy_templates_Reduction<dtype> red(this->Group::thisgroup);
  red[0].remoteRecv(msg);
}

template <class dtype> void
templates_Reduction<dtype>::Register(templates_ClientMsg *msg)
{
  cid = msg->cid;
  this->thisProxy;
  delete msg;
}

template <class dtype> void
templates_Reduction<dtype>::remoteRecv(templates_Single<dtype> *msg)
{
  data += msg->data;
  nreported++;
  if(nreported == CkNumPes()) {
    msg->data = data;
    CProxy_templates_Collector<dtype> col(cid);
    col.collect(msg);
    nreported = 0; data = 0;
  } else {
    delete msg;
  }
}

template <class dtype> 
templates_Collector<dtype>::templates_Collector(void)
{
  CProxy_templates_Reduction<dtype> red(templates_redid);
  if(CkMyPe()==0) {
    templates_ClientMsg *cmsg = new templates_ClientMsg(this->Chare::thishandle);
    red[0].Register(cmsg);
  }
  templates_Single<dtype> *m = new templates_Single<dtype>((dtype)(CkMyPe()+1));
  red[CkMyPe()].submit(m);
}

template <class dtype> void
templates_Collector<dtype>::collect(templates_Single<dtype> *msg)
{
  dtype data = msg->data;
  delete msg;
  if(data != (dtype) ((CkNumPes()*(CkNumPes()+1))/2)) {
    CkAbort("templates: test failed!\n");
  }
  megatest_finish();
}


template <class dtype> void
templates_Array<dtype>::remoteRecv(templates_Single<dtype> *msg) 
{
	data+= msg->data;
	delete msg;
}

template <class dtype> void 
templates_Array<dtype>::marshalled(int len,dtype *arr)
{
	for (int i=0;i<len;i++) data+=arr[i];
}

MEGATEST_REGISTER_TEST(templates,"milind",0)
