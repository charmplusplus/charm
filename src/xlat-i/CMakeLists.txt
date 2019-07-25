set(charmxi-cxx-sources xi-AstNode.C xi-BlockConstruct.C xi-Chare.C
    xi-Construct.C xi-Entry.C xi-Member.C xi-Message.C xi-Module.C xi-Parameter.C
    xi-SdagCollection.C xi-SdagConstruct.C xi-Template.C xi-Type.C xi-Value.C
    xi-grammar.tab.C xi-main.C xi-scan.C xi-symbol.C xi-util.C sdag/CEntry.C
    sdag/CParsedFile.C sdag/CSdagConstruct.C sdag/CStateVar.C sdag/sdag-globals.C
    sdag/constructs/Case.C sdag/constructs/CaseList.C sdag/constructs/Else.C
    sdag/constructs/For.C sdag/constructs/Forall.C sdag/constructs/If.C
    sdag/constructs/IntExpr.C sdag/constructs/OList.C sdag/constructs/Overlap.C
    sdag/constructs/SList.C sdag/constructs/SdagEntry.C sdag/constructs/Serial.C
    sdag/constructs/Template.C sdag/constructs/When.C sdag/constructs/While.C)

set(charmxi-h-sources xi-AstNode.h xi-BlockConstruct.h xi-Chare.h
    xi-Construct.h xi-Entry.h xi-Member.h xi-Message.h xi-Module.h xi-Parameter.h
    xi-SdagCollection.h xi-SdagConstruct.h xi-Template.h xi-Type.h xi-Value.h
    xi-grammar.tab.h xi-main.h xi-symbol.h xi-util.h sdag/CEntry.h
    sdag/CParsedFile.h sdag/CStateVar.h sdag/CToken.h sdag/EToken.h
    sdag/sdag-globals.h sdag/constructs/Case.h sdag/constructs/CaseList.h
    sdag/constructs/Constructs.h sdag/constructs/Else.h sdag/constructs/For.h
    sdag/constructs/Forall.h sdag/constructs/If.h sdag/constructs/IntExpr.h
    sdag/constructs/OList.h sdag/constructs/Overlap.h sdag/constructs/SList.h
    sdag/constructs/SdagEntry.h sdag/constructs/Serial.h
    sdag/constructs/Template.h sdag/constructs/When.h sdag/constructs/While.h)


add_executable(charmxi ${charmxi-cxx-sources} ${charmxi-h-sources})

target_include_directories(charmxi PRIVATE sdag .)

target_link_libraries(charmxi -host "-language c++")
