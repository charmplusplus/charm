#include "charm++.h" /* Has both CkPrintf and ckstream.h */


/// The one true output routine: write these n characters.
///  buf is always zero-terminated.
void CkOStreamBuf::myWrite(const char *buf,int n) {
  if (isErr==0)
    CkPrintf("%s",buf);
  else /*isErr==1*/
    CkError("%s",buf);
}

/// Write any buffered characters to the output.
int CkOStreamBuf::sync ()
{ 
  int n = pptr() - pbase();
  if (n!=0) { // Our output buffer is full-- flush it 
    buf[n]=0; //Zero-terminate our output buffer
    myWrite(pbase(), n);
    resetMyBuffer();
  }
  return 0;
}

/// Our buffer is full: handle it
int CkOStreamBuf::overflow (int ch)
{ 
  //Flush existing buffer:
  if (sync()) return EOF;
  
  if (ch!=EOF) 
  { //Put next character into buffer:
    *pptr()=ch;
    pbump(1);
  }
  return 0;
}


CpvDeclare(CkOutStream,_ckout);
CpvDeclare(CkErrStream,_ckerr);
CkInStream ckin;

void CkStreamInit(char **argv) {
	CpvInitialize(CkOutStream,_ckout);
	CpvInitialize(CkErrStream,_ckerr);
}
