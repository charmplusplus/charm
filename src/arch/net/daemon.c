/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <winsock2.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <process.h>
#include <time.h>

#include "sockRoutines.h"
#include "daemon.h"

/*If FACELESS is defined, the daemon logs runs to a file
(daemon.log) and doesn't show a DOS window.
Otherwise, the daemon logs things to its DOS window.
*/
/*#define FACELESS  /*<- sent in from the makefile*/

/*
Paste the environment string oldEnv just after dest.
This is a bit of a pain since windows environment strings
are double-null terminated.
  */
void envCat(char *dest,LPTSTR oldEnv)
{
  char *src=oldEnv;
  dest+=strlen(dest);//Advance to end of dest
  dest++;//Advance past terminating NULL character
  while ((*src)!='\0') {
    int adv=strlen(src)+1;//Length of newly-copied string plus NULL
    strcpy(dest,src);//Copy another environment string
    dest+=adv;//Advance past newly-copied string and NULL
    src+=adv;//Ditto for src
  }
  *dest='\0';//Paste on final terminating NULL character
  FreeEnvironmentStrings(oldEnv);
}

FILE *logfile=stdout;/*Status messages to standard output*/

void abort_writelog(int code,const char *msg) {
	fprintf(logfile,"Socket error %d-- %s!\n",code,msg);
	fclose(logfile);
	exit(3);
}

int startProgram(const char *exeName, const char *args, 
				const char *cwd, const char *env);

void goFaceless(void);

void main()
{
  unsigned int myPortNumber = DAEMON_IP_PORT;
  int myfd;
  
  int remotefd;         /* Remote Process Connecting */
  int remoteIP;         /* Remote Process's IP */
  unsigned int remotePortNumber; /* Port on which remote port is connecting */
  
  taskStruct task;      /* Information about the task to be performed */
  time_t curTime;

#ifdef FACELESS
  logfile=fopen("daemon.log","w+");
  if (logfile==NULL) /*Couldn't open log file*/
    logfile=stdout;
  else 
	goFaceless();
#endif
  
  curTime=time(NULL);
  fprintf(logfile,"Logfile for Windows Spawner Daemon for Converse programs\n"
	  "Run starting: %s\n\n",ctime(&curTime));fflush(logfile);
  
  skt_init();
  skt_set_abort(abort_writelog);
  
  /* Initialise "Listening Socket" */
  myfd=skt_server(&myPortNumber);
  
  while(1) {
    char *argLine; /* Argument list for called program */
    char statusCode;/*Status byte sent back to requestor*/
    
    fprintf(logfile,"\nListening for requests on port %d\n",myPortNumber);
	fflush(logfile);
    
    /* Accept & log a TCP connection from a client */
    remotefd=skt_accept(myfd, &remoteIP, &remotePortNumber); 
    
    curTime=time(NULL);
    fprintf(logfile,"Connection from IP 0x%08x, port %d"
	    "on %s",remoteIP,remotePortNumber,ctime(&curTime));
	fflush(logfile);
    
    /* Recv the task to be done */
    skt_recvN(remotefd, (BYTE *)&task, sizeof(task));
    task.pgm[DAEMON_MAXPATHLEN-1]=0; /*null terminate everything in sight*/
    task.cwd[DAEMON_MAXPATHLEN-1]=0;
    task.env[DAEMON_MAXENV-1]=0;

    /* Check magic number */
    if (ChMessageInt(task.magic)!=DAEMON_MAGIC) {
      fprintf(logfile,"************ SECURITY ALERT! ************\n"
	      "Received execution request with the wrong magic number 0x%08x!\n"
	      "This could indicate someone is trying to hack your system.\n\n\n",ChMessageInt(task.magic));
      fflush(logfile);
	  continue; /*DON'T execute this command (could be evil!)*/
    }
    
    /* Allocate memory for arguments*/
    argLine = (char *)malloc(sizeof(char) * (ChMessageInt(task.argLength)+1));
    
    /* Recv the command line*/
    skt_recvN(remotefd, (BYTE *)argLine, ChMessageInt(task.argLength));
    
    /*Add null terminator*/
    argLine[ChMessageInt(task.argLength)] = 0;
    
    fprintf(logfile,"Invoking '%s'\n"
	    "and environment '%s'\n"
	    "in '%s'\n",
	    task.pgm,task.env,task.cwd);fflush(logfile);
    
    /* Finally, create the process*/
    if(startProgram(task.pgm,argLine,task.cwd,task.env) == 0){
      /*Something went wrong!  Look up the Windows error code*/
      int error=GetLastError();
      statusCode=daemon_err2status(error);
      fprintf(logfile,"******************* ERROR *****************\n"
	      "Error in creating process!\n"
	      "Error code = %ld-- %s\n\n\n", error,
	      daemon_status2msg(statusCode));fflush(logfile);
    }
    else
      statusCode='G';//Status is good-- started program sucessfully
    /*Send status byte back to requestor*/
    skt_sendN(remotefd,(BYTE *)&statusCode,sizeof(char));

    /*Free recv'd arguments*/
	free(argLine);
  }
  
}

#ifdef _WIN32
int startProgram(const char *exeName, const char *args, 
				const char *cwd, const char *env)
{
  int ret;
  PROCESS_INFORMATION pi;         /* process Information for the process spawned */
  STARTUPINFO si={0};                 /* startup info for the process spawned */

  char environment[10000];/*Doubly-null terminated environment strings*/
  char cmdLine[10000];/*Program command line, including executable name*/
  if (strlen(exeName)+strlen(args) > 10000) 
	return 0; /*Command line too long.*/
  strcpy(cmdLine,exeName);
  strcat(cmdLine," ");
  strcat(cmdLine,args);

  /*Copy over the environment variables*/
  strcpy(environment,env);
  /*Paste all system environment strings after task.env */
  envCat(environment,GetEnvironmentStrings());
  
  /* Initialise the security attributes for the process 
     to be spawned */
  si.cb = sizeof(si);   

  ret = CreateProcess(NULL,	/* application name */
			    cmdLine,	/* command line */
			    NULL,/*&sa,*/							/* process SA */
			    NULL,/*&sa,*/							/* thread SA */
			    FALSE,							/* inherit flag */
#ifdef FACELESS
				CREATE_NEW_PROCESS_GROUP|DETACHED_PROCESS, 
#else
				CREATE_NEW_PROCESS_GROUP|CREATE_NEW_CONSOLE,
#endif
				/* creation flags */
			    environment,					/* environment block */
			    cwd,							/* working directory */
			    &si,							/* startup info */
			    &pi);
  return ret;
}

void goFaceless(void)
{
    printf("Switching to background mode...\n");
	fflush(stdout);
    sleep(2);/*Give user a chance to read message*/
    FreeConsole();
}

#else /*UNIX systems*/

# error "UNIX daemon not implemented."

#endif
