#include <jni.h>
#include "converse.h"
#include <stdlib.h>
#include <stdarg.h>

JavaVM *jvm;
JNIEnv *env;
jclass rtcls, cls, strClass, reg;
jmethodID mid;
char *appClassName;
  
JDK1_1InitArgs vm_args;

static void P(const char *format, ...) {
#if 0
  va_list args;
  int i;
  char str[1024];
  va_start(args, format);
  vsprintf(str, format, args);
  CmiPrintf("[%d] %s", CmiMyPe(), str);
#endif
}

void JavaInit(int argc, char **argv);

int main(int argc, char *argv[])
{
  JNI_GetDefaultJavaVMInitArgs(&vm_args);
  vm_args.classpath = ".:/home/kale/milind/mdJava:/expand1/software/jdk1.1/lib/classes.zip";
  JNI_CreateJavaVM(&jvm, &env, &vm_args);
  ConverseInit(argc, argv, JavaInit, 1, 1);
  JavaInit(argc, argv);
}

void JavaInit(int argc, char **argv) 
{

  P("Started jhost...\n");
  if(argc < 2) {
    CmiError("Usage: jhost AppClassName\n");
    ConverseExit();
    exit(1);
  } else {
    appClassName = argv[1];
  }
  
  rtcls = (*env)->FindClass(env, "parallel/PRuntime");
  P("Found class parallel/PRuntime\n");
  P("rtcls=%d\n", rtcls);
  reg = (*env)->FindClass(env, "RegisterAll");
  P("Found class RegisterAll=%d\n",reg);
  mid = (*env)->GetStaticMethodID(env,reg,"registerAll","()V");
  P("Found method registerAll=%d\n",mid);
  (*env)->CallStaticVoidMethod(env,reg, mid);
  P("Finished Registration\n");

  if (CmiMyPe() == 0) {
    int argvlen = argc - 2;
    int i;
    jobjectArray argvarray; 
    jstring nullString = (*env)->NewStringUTF(env," ");
    strClass = (*env)->FindClass(env,"java/lang/String");
    P("Found class java.lang.String, cls=%d\n", strClass);
    cls = (*env)->FindClass(env,appClassName);
    P("Found class %s, cls=%d\n", appClassName, cls);
    mid = (*env)->GetStaticMethodID(env,cls, "main", "([Ljava/lang/String;)V");
    P("Found method main=%d\n",mid);
    argvarray = (*env)->NewObjectArray(env,argvlen,strClass,nullString);
    P("Allocated argv array\n");
    for(i=0;i<argvlen;i++) {
      (*env)->SetObjectArrayElement(env,argvarray,i,
                                    (*env)->NewStringUTF(env,argv[i+2]));
    }
    P("Constructed array argv\n");
    (*env)->CallStaticVoidMethod(env,cls, mid, argvarray);
    P("Finished calling Main method..\n");
  }
  
  mid = (*env)->GetStaticMethodID(env,rtcls, "CStartScheduler", "()V");
  P("Found method startScheduler\n");
  (*env)->CallStaticVoidMethod(env,rtcls, mid);
  P("Out of scheduler...\n");
  
  (*jvm)->DestroyJavaVM(jvm);
  ConverseExit();
}

