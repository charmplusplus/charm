
tree grammar InitPUPCollector;

options {
    tokenVocab = Charj;
    ASTLabelType = CharjAST;
    filter = true;
}

@header {
    package charj.translator;
}

@members {
    Scope currentScope;
    ClassSymbol currentClass = null;
    boolean inMethod = false;
}

topdown
    :   enterClass
    |   enterMethod
    |   enterDefaultConstructor
    |   enterMigrationConstructor
    |   varDeclaration
    ;

bottomup
    :   exitMethod
    |   exitClass
    ;

classType
    :   CLASS
    |   CHARE
    |   GROUP
    |   NODEGROUP
    |   MAINCHARE
    |   ^(CHARE_ARRAY ARRAY_DIMENSION)
    ;

enterClass
    :   ^(TYPE classType IDENT
        (^('extends' .*))?
        (^('implements' .*))?
        (^((FUNCTION_METHOD_DECL | ENTRY_FUNCTION_DECL | PRIMITIVE_VAR_DECLARATION | DIVCON_METHOD_DECL |
            OBJECT_VAR_DECLARATION | CONSTRUCTOR_DECL | ENTRY_CONSTRUCTOR_DECL) .*))*)
        {
            currentClass = (ClassSymbol)$IDENT.def.type;
            currentClass.isMainChare = $classType.text.equals("mainchare");
        }
    ;

exitClass
    :   ^(TYPE classType IDENT
        (^('extends' .*))?
        (^('implements' .*))?
        (^((FUNCTION_METHOD_DECL | ENTRY_FUNCTION_DECL | PRIMITIVE_VAR_DECLARATION | DIVCON_METHOD_DECL |
            OBJECT_VAR_DECLARATION | CONSTRUCTOR_DECL | ENTRY_CONSTRUCTOR_DECL) .*))*)
        {
            currentClass = null;
        }
    ;

enterDefaultConstructor
    :   FORMAL_PARAM_LIST
        {
            if (($FORMAL_PARAM_LIST.hasParentOfType(CONSTRUCTOR_DECL) ||
                 $FORMAL_PARAM_LIST.hasParentOfType(ENTRY_CONSTRUCTOR_DECL)) && currentClass != null)

                    currentClass.hasDefaultCtor = true;
        }
    ;

enterMigrationConstructor
    :    ^(FORMAL_PARAM_LIST ^(FORMAL_PARAM_STD_DECL
                ^(MESSAGE_TYPE ^(QUALIFIED_TYPE_IDENT IDENT)) .
            ))
        {
            if (($FORMAL_PARAM_LIST.hasParentOfType(CONSTRUCTOR_DECL) ||
                 $FORMAL_PARAM_LIST.hasParentOfType(ENTRY_CONSTRUCTOR_DECL)) && currentClass != null)

                if($IDENT.text.equals("CkMigrateMessage")) 
                    currentClass.hasMigrationCtor = true;
                else if($IDENT.text.equals("CkArgMsg") && currentClass.isMainChare)
                    currentClass.hasDefaultCtor = true;
        }
    ;

enterMethod
    :   ^((FUNCTION_METHOD_DECL | ENTRY_FUNCTION_DECL
            | CONSTRUCTOR_DECL | ENTRY_CONSTRUCTOR_DECL) .*)
        {
            inMethod = true;
        }
    ;

exitMethod
    :   ^((FUNCTION_METHOD_DECL | ENTRY_FUNCTION_DECL
            | CONSTRUCTOR_DECL | ENTRY_CONSTRUCTOR_DECL) .*)
        {
            inMethod = false;
        }
    ;

varDeclaration
    :   ^(VAR_DECLARATOR ^(IDENT .*) (expr=.)? )
        {

            if (!inMethod && currentClass != null && $expr != null) {
                currentClass.initializers.add(new VariableInitializer($expr, $IDENT));
            }

            if (!inMethod && currentClass != null) {
				currentClass.varsToPup.add($IDENT);
				if(!($IDENT.symbolType instanceof ProxyType || $IDENT.symbolType instanceof ProxySectionType))
					currentClass.pupInitializers.add(new VariableInitializer($expr, $IDENT));
            }
        }
    ;


