/*Exactly like the system() call, but signal-safe, so 
  it will work from the (non-netpoll) net- version
  instead of just returning EINTR on the first SIGIO.
  
  This code is #included by both ckdll.C and the configure script
  (which tests if it will work).
*/
#include <stdlib.h> /* for exit */
#include <unistd.h> /* for execv */
#include <sys/wait.h> /* for sys/wait.h */
#include <errno.h> 
int CkSystem (const char *command) {
   int pid, status;

   pid = fork();
   if (pid == -1)
       return -1;
   if (pid == 0) { /*Child: exec the shell*/
               char *argv[4];
               argv[0] = strdup("sh");
               argv[1] = strdup("-c");
               argv[2] = (char *)command;
               argv[3] = 0;
               execv("/bin/sh", argv);
               exit(127);
   }
   do { /*Parent: wait for child*/
     if (waitpid(pid, &status, 0) == -1) {
       if (errno != EINTR)
         return -1;
     } 
     else /*waitpid succeeded*/
        return status;
   } while(1);
}
