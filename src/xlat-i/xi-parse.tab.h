#define BOC 257
#define CHARE 258
#define ENTRY 259
#define MESSAGE 260
#define PACKMESSAGE 261
#define READONLY 262
#define STACKSIZE 263
#define TABLE 264
#define THREADED 265
#define VARSIZE 266
#define EXTERN 267
#define IDENTIFIER 268
#define NUMBER 269
typedef union {
	char *strval;
	int intval;
} YYSTYPE;
extern YYSTYPE yylval;
