#define BOC 257
#define CHARE 258
#define ENTRY 259
#define MESSAGE 260
#define PACKMESSAGE 261
#define READONLY 262
#define STACKSIZE 263
#define TABLE 264
#define THREADED 265
#define EXTERN 266
#define IDENTIFIER 267
#define NUMBER 268
typedef union {
	char *strval;
	int intval;
} YYSTYPE;
extern YYSTYPE yylval;
