#define BOC 257
#define CHARE 258
#define ENTRY 259
#define MESSAGE 260
#define READONLY 261
#define TABLE 262
#define THREADED 263
#define EXTERN 264
#define IDENTIFIER 265
typedef union {
	char *strval;
	int intval;
} YYSTYPE;
extern YYSTYPE yylval;
