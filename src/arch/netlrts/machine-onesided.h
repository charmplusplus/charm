#ifndef NETLRTS_MACHINE_ONESIDED_H
#define NETLRTS_MACHINE_ONESIDED_H

void LrtsSetRdmaBufferInfo(void *info, const void *ptr, int size, unsigned short int mode) {}

void LrtsIssueRget(NcpyOperationInfo *ncpyOpInfo){
  CmiAbort("Should never reach here!");
}

void LrtsIssueRput(NcpyOperationInfo *ncpyOpInfo){
  CmiAbort("Should never reach here!");
}

void LrtsDeregisterMem(const void *ptr, void *info, int pe, unsigned short int mode) {}

#endif /* NETLRTS_MACHINE_ONESIDED_H */
