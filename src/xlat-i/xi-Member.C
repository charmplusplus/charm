#include "xi-Chare.h"
#include "xi-Member.h"
#include "xi-Value.h"

namespace xi {

XStr makeIdent(const XStr& in);

extern int fortranMode;

// Return a templated proxy declaration string for
// this Member's container with the given return type, e.g.
// template<int N,class foo> void CProxy_bar<N,foo>
// Works with non-templated Chares as well.
XStr Member::makeDecl(const XStr& returnType, int forProxy, bool isStatic, XStr fwdStr) {
  XStr str;

  if (container->isTemplated()) str << container->tspec(false) << "\n";
  const bool doFwd = fwdStr != "";
  if (tspec || doFwd) {
    str << "template <";
    if (tspec) {
      tspec->genLong(str, false);
      if (doFwd)
        str << ", ";
    }
    if (doFwd)
      str << fwdStr;
    str << ">\n";
  }
  if (isStatic) str << "static ";
  str << returnType << " ";
  if (forProxy == 1)
    str << container->proxyName();
  else if (forProxy == 0)
    str << container->indexName();
  else
    str << container->sectionName();
  return str;
}

void Readonly::print(XStr& str) {
  if (external) str << "extern ";
  str << "readonly ";
  if (msg) str << "message ";
  type->print(str);
  if (msg)
    str << " *";
  else
    str << " ";
  str << name;
  if (dims) dims->print(str);
  str << ";\n";
}

// Method to declare a CkNcpyBuffer for arrays
void Readonly::genZCDeclForArrays(XStr& str) {
  str << "    int regMode = CK_BUFFER_REG;\n";
  str << "    if(CkNumNodes() == 1)\n";
  str << "      regMode = CK_BUFFER_UNREG;\n"; // No point of registration when there are no ZC ops being performed
  str << "    CkNcpyBuffer myBuffer(& "<< qName();
  dims->printZeros(str);
  str <<", (";
  dims->printValueProduct(str);
  str <<" * sizeof(";
  type->print(str);
  str <<")), regMode);\n";
}

// Method to declare a CkNcpyBuffer for std::vector
void Readonly::genZCDeclForVectors(XStr& str, NamedType *nType) {
  str << "      int regMode = CK_BUFFER_REG;\n";
  str << "      if(CkNumNodes() == 1)\n";
  str << "        regMode = CK_BUFFER_UNREG;\n"; // No point of registration when there are no ZC ops being performed
  str << "      CkNcpyBuffer myBuffer("<< qName();
  str << ".data()";
  str << ", ";
  str << "sizeof(" << nType->getTparams();
  str << ") * "<< qName() <<".size()";
  str <<", regMode);\n";
}

void Readonly::genDefs(XStr& str) {
  str << "/* DEFS: ";
  print(str);
  str << " */\n";
  if (!container && !strchr(name, ':')) {
    str << "extern ";
    type->print(str);
    if (msg) str << "*";
    str << " " << name;
    if (dims) dims->print(str);
    str << ";\n";
  }

  if (!msg) {  // Generate a pup for this readonly
    templateGuardBegin(false, str);
    str << "extern \"C\" void __xlater_roPup_" << makeIdent(qName());
    str << "(void *_impl_pup_er) {\n";
    str << "  PUP::er &_impl_p=*(PUP::er *)_impl_pup_er;\n";
    if (dims) {
      // Setting CMK_ONESIDED_RO_DISABLE provides a compile time switch to turn off RO ZC Bcast
      str <<"#if !CMK_ONESIDED_RO_DISABLE && CMK_ONESIDED_IMPL\n";
      // Conditional to selectively pup ncpy buffers
      str <<"  if(";
      dims->printValueProduct(str);
      str <<" * sizeof(";
      type->print(str);
      str << ") >= CMK_ONESIDED_RO_THRESHOLD) {\n";

      str <<"    if(_impl_p.isSizing()) {\n";
      str <<"      CkNcpyBuffer myBuffer;\n";
      str <<"      _impl_p|myBuffer;\n";
      str <<"      readonlyUpdateNumops();\n";
      str <<"    }\n";
      str <<"    if(_impl_p.isPacking()) {\n";
      genZCDeclForArrays(str);
      str <<"      _impl_p|myBuffer;\n";
      str <<"      if(CkNumNodes() > 1)\n";
      str <<"        readonlyCreateOnSource(myBuffer);\n";
      str <<"      PUP::toMem &_impl_p_toMem = *(PUP::toMem *)_impl_pup_er;\n";
      str <<"      envelope *env = UsrToEnv(_impl_p_toMem.get_orig_pointer());\n";
      str <<"      CMI_ZC_MSGTYPE(env) = CMK_ZC_BCAST_SEND_MSG;\n";
      str <<"    }\n";
      str <<"    if(_impl_p.isUnpacking()) {\n";
      genZCDeclForArrays(str);
      str <<"      PUP::fromMem &_impl_p_fromMem = *(PUP::fromMem *)_impl_pup_er;\n";
      str <<"      char *ptr = _impl_p_fromMem.get_current_pointer();\n";
      str <<"      PUP::toMem _impl_p_toMem = (PUP::toMem)((void *)ptr);\n";
      str <<"      envelope *env = UsrToEnv(_impl_p_fromMem.get_orig_pointer());\n";
      str <<"      CkNcpyBuffer srcBuffer;\n";
      str <<"      _impl_p|srcBuffer;\n";
      str <<"      _impl_p_toMem|myBuffer;\n";
      str <<"      readonlyGet(srcBuffer, myBuffer, (void *)env);\n";
      str <<"    }\n";
      str <<"  } else\n";
      str <<"#endif\n";
      str << "  {\n";
      str << "  _impl_p(&" << qName();
      dims->printZeros(str);
      str << ", (";
      dims->printValueProduct(str);
      str << ") );\n";
      str << "  }\n";
    } else {
      if(strcmp("vector",type->getBaseName()) == 0) {
        // Setting CMK_ONESIDED_RO_DISABLE provides a compile time switch to turn off RO ZC Bcast
        str <<"#if !CMK_ONESIDED_RO_DISABLE && CMK_ONESIDED_IMPL\n";
        NamedType *nType = (NamedType *)type;

        // Determine if the vector is going to be using ZC bcast
        str <<"  bool vecUsesZC = false;\n";
        str <<"  if(_impl_p.isPacking() || _impl_p.isSizing()) {\n";
        // Add conditional to selectively pup ncpy buffers
        str <<"    vecUsesZC = (" << qName() <<".size()";
        str <<" * sizeof(" << nType->getTparams() <<")";
        str <<" >= CMK_ONESIDED_RO_THRESHOLD);\n";
        str <<"  }\n";

        str <<"  _impl_p|vecUsesZC;\n";

        str <<"  if(vecUsesZC) {\n";
        str <<"    if(_impl_p.isSizing()) {\n";
        str <<"      CkNcpyBuffer myBuffer;\n";
        str <<"      _impl_p|myBuffer;\n";
        str <<"      readonlyUpdateNumops();\n";
        str <<"    }\n";
        str <<"    if(_impl_p.isPacking()) {\n";
        genZCDeclForVectors(str, nType);
        str <<"      _impl_p|myBuffer;\n";
        str <<"      if(CkNumNodes() > 1)\n";
        str <<"        readonlyCreateOnSource(myBuffer);\n";
        str <<"      PUP::toMem &_impl_p_toMem_orig = *(PUP::toMem *)_impl_pup_er;\n";
        str <<"      envelope *env = UsrToEnv(_impl_p_toMem_orig.get_orig_pointer());\n";
        str <<"      CMI_ZC_MSGTYPE(env) = CMK_ZC_BCAST_SEND_MSG;\n";
        str <<"    }\n";
        str <<"    if(_impl_p.isUnpacking()) {\n";
        str <<"      PUP::fromMem &_impl_p_fromMem = *(PUP::fromMem *)_impl_pup_er;\n";
        str <<"      envelope *env = UsrToEnv(_impl_p_fromMem.get_orig_pointer());\n";
        str <<"      char *ptr = _impl_p_fromMem.get_current_pointer();\n";
        str <<"      PUP::toMem _impl_p_toMem = (PUP::toMem)((void *)ptr);\n";
        str <<"      CkNcpyBuffer srcBuffer;\n";
        str <<"      _impl_p|srcBuffer;\n";
        str <<"      size_t nElem = ";
        str <<"(srcBuffer.cnt)/sizeof("<<nType->getTparams()<<");\n";
        str <<"      " << qName() <<".resize(nElem);\n";
        str <<"      " << qName() <<".shrink_to_fit();\n";
        str <<"      CkNcpyBuffer myBuffer("<< qName() << ".data()";
        str <<", " << "srcBuffer.cnt, CK_BUFFER_REG);\n";
        str <<"      _impl_p_toMem|myBuffer;\n";
        str <<"      readonlyGet(srcBuffer, myBuffer, (void *)env);\n";
        str <<"    }\n";
        str <<"  } else\n";
        str <<"#endif\n";
        str <<"  {\n";
        str <<"    _impl_p|" << qName() << ";\n";
        str <<"  }\n";
      } else {
        str <<"  _impl_p|" << qName() << ";\n";
      }
    }
    str << "}\n";
    templateGuardEnd(str);
  }

  if (fortranMode) {
    str << "extern \"C\" void " << fortranify("set_", name) << "(int *n) { " << name
        << " = *n; }\n";
    str << "extern \"C\" void " << fortranify("get_", name) << "(int *n) { *n = " << name
        << "; }\n";
  }
}

void Readonly::genReg(XStr& str) {
  if (external) return;
  if (msg) {
    if (dims) XLAT_ERROR_NOCOL("readonly message cannot be an array", line);
    str << "  CkRegisterReadonlyMsg(\"" << qName() << "\",\"" << type << "\",";
    str << "(void **)&" << qName() << ");\n";
  } else {
    str << "  CkRegisterReadonly(\"" << qName() << "\",\"" << type << "\",";
    str << "sizeof(" << qName() << "),(void *) &" << qName() << ",";
    str << "__xlater_roPup_" << makeIdent(qName()) << ");\n";
  }
}

XStr Readonly::qName(void) const { /*Return fully qualified name*/
  XStr ret;
  if (container) ret << container->baseName() << "::";
  ret << name;
  return ret;
}

Readonly::Readonly(int l, Type* t, const char* n, ValueList* d, int m)
    : msg(m), type(t), name(n) {
  line = l;
  dims = d;
  setChare(0);
}

void Readonly::genDecls(XStr& str) {
  str << "/* DECLS: ";
  print(str);
  str << " */\n";
}

void Readonly::genIndexDecls(XStr& str) {
  str << "/* DECLS: ";
  print(str);
  str << " */\n";
}

void InitCall::setAccel() { isAccelFlag = 1; }
void InitCall::clearAccel() { isAccelFlag = 0; }
int InitCall::isAccel() { return isAccelFlag; }

/***************** PUP::able support **************/
PUPableClass::PUPableClass(int l, NamedType* type_, PUPableClass* next_)
    : type(type_), next(next_) {
  line = l;
  setChare(0);
}
void PUPableClass::print(XStr& str) {
  str << "  PUPable " << type << ";\n";
  if (next) next->print(str);
}
void PUPableClass::genDefs(XStr& str) {
  bool isTemplate = type->isTemplated();
  templateGuardBegin(isTemplate, str);
  if (isTemplate) {
    str << "  #define _CHARMXI_CLASS_NAME " << type << "\n";
    str << "  PUPable_def_template(_CHARMXI_CLASS_NAME)\n";
    str << "  #undef _CHARMXI_CLASS_NAME\n";
  } else {
    str << "  PUPable_def(" << type << ")\n";
  }
  templateGuardEnd(str);
  if (next) next->genDefs(str);
}
void PUPableClass::genReg(XStr& str) {
  if (type->isTemplated()) {
    str << "      #define _CHARMXI_CLASS_NAME " << type << "\n";
    str << "      PUPable_reg2(_CHARMXI_CLASS_NAME,\"" << type << "\");\n";
    str << "      #undef _CHARMXI_CLASS_NAME\n";
  } else {
    str << "      PUPable_reg(" << type << ");\n";
  }
  if (next) next->genReg(str);
}

// DMK - Accel Support
int PUPableClass::genAccels_spe_c_funcBodies(XStr& str) {
  int rtn = 0;
  if (next) {
    rtn += next->genAccels_spe_c_funcBodies(str);
  }
  return rtn;
}

void PUPableClass::genAccels_spe_c_regFuncs(XStr& str) {
  if (next) {
    next->genAccels_spe_c_regFuncs(str);
  }
}

void PUPableClass::genAccels_spe_c_callInits(XStr& str) {
  if (next) {
    next->genAccels_spe_c_callInits(str);
  }
}

void PUPableClass::genAccels_spe_h_includes(XStr& str) {
  if (next) {
    next->genAccels_spe_h_includes(str);
  }
}

void PUPableClass::genAccels_spe_h_fiCountDefs(XStr& str) {
  if (next) {
    next->genAccels_spe_h_fiCountDefs(str);
  }
}

void PUPableClass::genAccels_ppe_c_regFuncs(XStr& str) {
  if (next) {
    next->genAccels_ppe_c_regFuncs(str);
  }
}

/***************** InitCall **************/
InitCall::InitCall(int l, const char* n, int nodeCall) : name(n) {
  line = l;
  setChare(0);
  isNodeCall = nodeCall;

  // DMK - Accel Support
  isAccelFlag = 0;
}
void InitCall::print(XStr& str) { str << "  initcall void " << name << "(void);\n"; }
void InitCall::genReg(XStr& str) {
  str << "  _registerInitCall(";
  if (container) str << container->baseName() << "::";
  str << name;
  str << "," << isNodeCall << ");\n";
}

void InitCall::genAccels_spe_c_callInits(XStr& str) {
  if (isAccel()) {
    str << "    " << name << "();\n";
  }
}

/***************** Include support **************/
IncludeFile::IncludeFile(int l, const char* n) : name(n) {
  line = l;
  setChare(0);
}
void IncludeFile::print(XStr& str) { str << "  include " << name << ";\n"; }
void IncludeFile::genDecls(XStr& str) { str << "#include " << name << "\n"; }

/***************** normal extern C Class support **************/
ClassDeclaration::ClassDeclaration(int l, const char* n) : name(n) {
  line = l;
  setChare(0);
}
void ClassDeclaration::print(XStr& str) { str << "  class " << name << ";\n"; }
void ClassDeclaration::genDecls(XStr& str) { str << "class " << name << ";\n"; }

// Turn this string into a valid identifier
XStr makeIdent(const XStr& in) {
  XStr ret;
  const char* i = in.get_string_const();
  while (*i != 0) {
    // Quote all "special" characters
    if (*i == ':')
      ret << "_QColon_";
    else if (*i == ' ')
      ret << "_QSpace_";
    else if (*i == '+')
      ret << "_QPlus_";
    else if (*i == '-')
      ret << "_QMinus_";
    else if (*i == '*')
      ret << "_QTimes_";
    else if (*i == '/')
      ret << "_QSlash_";
    else if (*i == '%')
      ret << "_QPercent_";
    else if (*i == '&')
      ret << "_QAmpersand_";
    else if (*i == '.')
      ret << "_QDot_";
    else if (*i == ',')
      ret << "_QComma_";
    else if (*i == '\'')
      ret << "_QSQuote_";
    else if (*i == '\"')
      ret << "_QQuote_";
    else if (*i == '(')
      ret << "_QLparen_";
    else if (*i == ')')
      ret << "_QRparen_";
    else if (*i == '<')
      ret << "_QLess_";
    else if (*i == '>')
      ret << "_QGreater_";
    else if (*i == '{')
      ret << "_QLbrace_";
    else if (*i == '}')
      ret << "_QRbrace_";
    else
      ret << *i;  // Copy character unmodified
    i++;          // Advance to next
  }
  return ret;
}

}  // namespace xi
