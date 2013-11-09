#define DAEMON_IP_PORT 12396

/*This is the structure that is passed to the Daemon when
you want it to start a process.
*/
#define DAEMON_MAXPATHLEN 512
#define DAEMON_MAXENV 256
typedef struct {
	char pgm[DAEMON_MAXPATHLEN]; /*Name of executable to run (no ".exe" needed)*/
	char cwd[DAEMON_MAXPATHLEN];/*Directory in which to start program*/

        char env[DAEMON_MAXENV];/*Environment variables*/

#define DAEMON_MAGIC 0x7AF2893C
	ChMessageInt_t magic;/*Magic number for daemon (a weak security measure)*/
	/*Length (bytes) of the program arguments to follow*/
	ChMessageInt_t argLength;
} taskStruct;

/*This table is used to look up the Windows error code
returned by CreateProcess and convert it into a "status code"
character (returned to conv-host) and an error message
(logged and printed by conv-host).
*/
#ifndef _WIN32
/*UNIX equivalents to Win32 errors*/
#define ERROR_FILE_NOT_FOUND EACCES
#define ERROR_NOT_ENOUGH_MEMORY ENOMEM
#define ERROR_OUTOFMEMORY E2BIG
#define ERROR_ACCESS_DENIED -1000 /*No equivalent*/
#define ERROR_SHARING_VIOLATION -1000
#define ERROR_BAD_EXE_FORMAT ENOEXEC
#endif
const static struct {
				int errorCode;
				char statusCode;/*'G'->sucess; all others failure*/
				const char *reason;
} daemon_errtab[]= {
				{ERROR_FILE_NOT_FOUND,'F',"executable not found."},
				{-1,'D',"directory not found."},
				{ERROR_NOT_ENOUGH_MEMORY,'M',"not enough memory."},
				{ERROR_OUTOFMEMORY,'M',"not enough memory."},
				{ERROR_ACCESS_DENIED,'A',"access denied."},
				{ERROR_SHARING_VIOLATION,'S',"sharing violation."},
				{ERROR_BAD_EXE_FORMAT,'E',"not an executable."},
				{-1,'N',"could not contact daemon-- is it running?"},
				{0,0,NULL}
};

/*Convert a windows error code into a status character*/
char daemon_err2status(int err)
{
	int i;
	for (i=0;daemon_errtab[i].reason;i++)
		if (daemon_errtab[i].errorCode==err)
			return daemon_errtab[i].statusCode;
	return 'U';/*If it's not in the table, it's an unknown error*/
}

/*Convert a status character into a human-readable error code*/
const char *daemon_status2msg(char statusCode)
{
	int i;
	for (i=0;daemon_errtab[i].reason;i++)
		if (daemon_errtab[i].statusCode==statusCode)
			return daemon_errtab[i].reason;
	return "unknown error.";
}





