/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <winsock2.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <process.h>
#include <time.h>

#include "daemon.h"
#include "sockRoutines.h"
#include "SocketReadAndWrite.h"

/*
Paste the environment string oldEnv just after dest.
  */
void envCat(char *dest,LPTSTR oldEnv)
{
	char *src=oldEnv;
	dest+=strlen(dest);//Advance to end of dest
	dest++;//Advance past terminating NULL character
	while ((*src)!='\0')
	{
		int adv=strlen(src)+1;//Length of newly-copied string plus NULL
		strcpy(dest,src);//Copy another environment string
		dest+=adv;//Advance past newly-copied string and NULL
		src+=adv;//Ditto for src
	}
	*dest='\0';//Paste on final terminating NULL character
	FreeEnvironmentStrings(oldEnv);
}


void cleanup(void)
{
   WSACleanup();
}

void main()
{
	SECURITY_ATTRIBUTES sa;         /* Security Attributes for the Thread/process spawned */
	PROCESS_INFORMATION pi;         /* process Information for the process spawned */
	STARTUPINFO si={0};                 /* startup info for the process spawned */
	int fStatus;                    /* Result of CreateProcess call */ 
	WORD wVersionRequested = MAKEWORD(2,0);
	WSADATA wsaData;

	int myPortNumber = DAEMON_IP_PORT;
	int myIP;
	int myfd;

	int remotefd;         /* Remote Process Connecting */
	int remoteIP;         /* Remote Process's IP */
	int remotePortNumber; /* Port on which remote port is connecting */

	taskStruct task;      /* Information about the task to be performed */
	int nStatus;          /* Result of any Socket calls */
	
	time_t curTime;

	FILE *logfile=stdout;/*Initially, log status messages to standard output*/

	curTime=time(NULL);
	fprintf(logfile,"Logfile for Windows Spawner Daemon for Converse programs\n"
			"Run starting: %s\n\n",ctime(&curTime));

	/* Initialise the WinSock System */
	WSAStartup(wVersionRequested,&wsaData);

	/* Initialise "Listening Socket" */
	skt_server(&myIP, &myPortNumber, &myfd);

	/* Initialise the security attributes for the process 
	   to be spawned */
	sa.nLength = sizeof(SECURITY_ATTRIBUTES);
	sa.lpSecurityDescriptor = NULL;
	sa.bInheritHandle = FALSE;

	si.cb = sizeof(si); 
	atexit(cleanup);    

	while(1){
		char *argLine; /* Argument list for called program */
		char cmdLine[1000];/*Program command line, including executable name*/

		char environment[20000];/*Doubly-null terminated environment string*/
		char statusCode;/*Status byte sent back to requestor*/

		fprintf(logfile,"Listening for requests on port %d\n",myPortNumber);

		/* Accept & log a TCP connection from a client */
		skt_accept(myfd, &remoteIP, &remotePortNumber, &remotefd); 

		curTime=time(NULL);
		fprintf(logfile,"Connection accepted from IP 0x%08x, port %d\n"
			"on %s",remoteIP,remotePortNumber,ctime(&curTime));

		/* Recv the task to be done */
		nStatus = RecvSocketN(remotefd, (BYTE *)&task, sizeof(task));
		if(nStatus == SOCKET_ERROR){
			perror("socket recv");
			break;
		}
		/* Check magic number */
		if (task.magic!=DAEMON_MAGIC)
		{
			fprintf(logfile,"************ SECURITY ALERT! ************\n"
					"Received execution request with the wrong magic number 0x%08x!\n"
					"This could indicate someone is trying to hack your system.\n\n\n",task.magic);
			continue;
		}
		
		/* Allocate memory for arguments*/
		argLine = (char *)malloc(sizeof(char) * (task.argLength+1));

		/* Recv the command line*/
		nStatus = RecvSocketN(remotefd, (BYTE *)argLine, task.argLength);
		if(nStatus == SOCKET_ERROR){
			perror("socket recv");
			break;
		}
		/*Add null terminator*/
		argLine[task.argLength] = 0;

		/*Combine task name and arguments*/
		strcpy(cmdLine,task.pgm);
		strcat(cmdLine,argLine);
		free((void *)argLine);
		
		/*Convert task info. into NETSTART environment variable*/
		sprintf(environment,"NETSTART=%d %d %d %d %d %d %d %d %d ",
					task.Cmi_numnodes, 
					task.Cmi_mynode,
					task.Cmi_nodestart,
					task.Cmi_mynodesize,
					task.Cmi_numpes,
					task.Cmi_self_IP,
					task.Cmi_host_IP,
					task.Cmi_host_port,
					task.Cmi_host_pid/*Windows environment strings are double-terminated*/
					);
		/*Paste all the old environment strings in after the NETSTART*/
		envCat(environment,GetEnvironmentStrings());

		fprintf(logfile,"Invoking process '%s' \n"
			"with command line '%s'\n"
			"and environment '%s'\n"
			"in working directory '%s'\n",
			task.pgm, cmdLine,environment,task.cwd);


		/* Finally, create the process*/
		fStatus = CreateProcess(NULL,	/* application name */
							cmdLine,	/* command line */
							NULL,/*&sa,*/							/* process SA */
							NULL,/*&sa,*/							/* thread SA */
							FALSE,							/* inherit flag */
							CREATE_NEW_CONSOLE|CREATE_NEW_PROCESS_GROUP, /* creation flags */
							environment,					/* environment block */
							task.cwd,							/* working directory */
							&si,							/* startup info */
							&pi);


		if(fStatus == 0){
			/*Something went wrong!  Look up the Windows error code*/
			int error=GetLastError();
			statusCode=daemon_err2status(error);
			fprintf(logfile,"******************* ERROR *****************\n"
				"Error in creating process!\n"
				"Error code = %ld-- %s\n\n\n", error,
				daemon_status2msg(statusCode));
		}
		else
		{
			fprintf(logfile,"Process created sucessfully\n\n\n");
			statusCode='G';//Status is good-- started program sucessfully
		}
		/*Send status byte back to requestor*/
		SendSocketN(remotefd,(BYTE *)&statusCode,sizeof(char));
	}

}

