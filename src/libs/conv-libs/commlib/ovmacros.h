#define OBAlloc(kptr, ksize) {\
  if (OBFreeList) {\
	kptr=OBFreeList;\
	OBFreeList=OBFreeList->next;\
  }\
  else {\
	 kptr=(OverlapBuffer *)CmiAlloc(ksize);\
  }\
}

#define OBFree(ktmp) {\
  ktmp->next=OBFreeList;\
  OBFreeList=ktmp;\
}

#define ORAlloc(kptr, ksize) {\
  if (ORFreeList) {\
	kptr=ORFreeList;\
	ORFreeList=ORFreeList->next;\
  }\
  else {\
	 kptr=(OverlapRecvBuffer *)CmiAlloc(ksize);\
  }\
}

#define ORFree(ktmp) {\
  ktmp->next=ORFreeList;\
  ORFreeList=ktmp;\
}

#define OPAlloc(kptr, ksize) {\
  if (OPFreeList) {\
	kptr=OPFreeList;\
	OPFreeList=OPFreeList->next;\
  }\
  else {\
	 kptr=(OverlapProcBuffer *)CmiAlloc(ksize);\
  }\
}

#define OPFree(ktmp) {\
  ktmp->next=OPFreeList;\
  OPFreeList=ktmp;\
}

#define ODAlloc(kptr, ksize) {\
  if (ODFreeList) {\
	kptr=ODFreeList;\
	ODFreeList=ODFreeList->next;\
  }\
  else {\
	 kptr=(OverlapDummyBuffer *)CmiAlloc(ksize);\
  }\
}

#define ODFree(ktmp) {\
  ktmp->next=ODFreeList;\
  ODFreeList=ktmp;\
}
