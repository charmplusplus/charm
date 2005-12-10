/*
unix2nt_cc  -c createlink.cpp -o createlink.o -D_WIN32_WINNT=0x0500
unix2nt_cc createlink.o
*/

#include <windows.h>
#include <stdio.h>

int main(int argc, char **argv)
{
  if (argc < 2) {
    printf("%s srcFile destFile\n", argv[0]);
    exit(1);
  }
  char *src = argv[1];
  char *dest = argv[2];
  bool fCreatedLink  = CreateHardLink(dest, src, NULL);
  if (!fCreatedLink) {
    int err=GetLastError();
    printf("CreateHardLink %s => %s failed errno=%d, WSAerr=%d\n", src, dest, errno, err);
    return 1;
  }
  else {
    exit(0);
  }
  return 0;
}

