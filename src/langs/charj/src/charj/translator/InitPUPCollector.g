
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
    boolean inBlock = false;
}

topdown
    :   enterClass
    |   enterBlock
    |   enterDefaultConstructor
    |   enterMigrationConstructor
    |   varDeclaration
    ;

bottomup
    :   exitBlock
    ;

enterClass
    :   ^(TYPE .* IDENT
            (^('extends' .*))?
            (^('implements' .*))?
            (^((FUNCTION_METHOD_DECL | ENTRY_FUNCTION_DECL | PRIMITIVE_VAR_DECLARATION |
                OBJECT_VAR_DECLARATION | CONSTRUCTOR_DECL | ENTRY_CONSTRUCTOR_DECL) .*))*)
        {
            currentClass = (ClassSymbol)$IDENT.def.type;
        }
    ;

enterDefaultConstructor
    :    ^((CONSTRUCTOR_DECL | ENTRY_CONSTRUCTOR_DECL)
            (^(MODIFIER_LIST .*))? .
            FORMAL_PARAM_LIST ^(BLOCK .*))
        {
            if (currentClass != null) {
                currentClass.hasDefaultConstructor = true;
            }
        }
    ;

enterBlock 
    :   ^(BLOCK .*)
        {
            inBlock = true;
        }
    ;

exitBlock 
    :   ^(BLOCK .*)
        {
            inBlock = false;
        }
    ;

varDeclaration
    :   ^((PRIMITIVE_VAR_DECLARATION | OBJECT_VAR_DECLARATION)
            (^(MODIFIER_LIST .*))? .
            ^(VAR_DECLARATOR_LIST (^(VAR_DECLARATOR ^(IDENT .*) .*)
            {
                if (!inBlock && currentClass != null) {
                    currentClass.varsToPup.add($IDENT);
                }
            }
                )+))
    |   ^(FORMAL_PARAM_STD_DECL (^(MODIFIER_LIST .+))? . ^(IDENT .*))
        {
            
        }
    ;


