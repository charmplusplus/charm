typedef union {
  ModuleList *modlist;
  Module *module;
  ConstructList *conslist;
  Construct *construct;
  TParam *tparam;
  TParamList *tparlist;
  Type *type;
  PtrType *ptype;
  NamedType *ntype;
  FuncType *ftype;
  Readonly *readonly;
  Message *message;
  Chare *chare;
  Entry *entry;
  Parameter *pname;
  ParamList *plist;
  Template *templat;
  TypeList *typelist;
  MemberList *mbrlist;
  Member *member;
  TVar *tvar;
  TVarList *tvarlist;
  Value *val;
  ValueList *vallist;
  MsgVar *mv;
  MsgVarList *mvlist;
  char *strval;
  int intval;
  Chare::attrib_t cattr;
} YYSTYPE;
#define	MODULE	257
#define	MAINMODULE	258
#define	EXTERN	259
#define	READONLY	260
#define	INITCALL	261
#define	CHARE	262
#define	MAINCHARE	263
#define	GROUP	264
#define	NODEGROUP	265
#define	ARRAY	266
#define	MESSAGE	267
#define	CLASS	268
#define	STACKSIZE	269
#define	THREADED	270
#define	TEMPLATE	271
#define	SYNC	272
#define	EXCLUSIVE	273
#define	IMMEDIATE	274
#define	VIRTUAL	275
#define	MIGRATABLE	276
#define	CREATEHERE	277
#define	CREATEHOME	278
#define	VOID	279
#define	CONST	280
#define	PACKED	281
#define	VARSIZE	282
#define	ENTRY	283
#define	IDENT	284
#define	NUMBER	285
#define	LITERAL	286
#define	CPROGRAM	287
#define	INT	288
#define	LONG	289
#define	SHORT	290
#define	CHAR	291
#define	FLOAT	292
#define	DOUBLE	293
#define	UNSIGNED	294


extern YYSTYPE yylval;
