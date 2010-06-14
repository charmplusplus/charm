
tree grammar SymbolDefiner;

options {
    tokenVocab = Charj;
    ASTLabelType = CharjAST;
    filter = true;
}

@header {
package charj.translator;
}

@members {
    SymbolTable symtab;
    Scope currentScope;
    ClassSymbol currentClass;

    public SymbolDefiner(TreeNodeStream input, SymbolTable symtab) {
        this(input);
        this.symtab = symtab;
        this.currentScope = symtab.getDefaultPkg();
        System.out.println(currentScope);
    }
}


topdown
    :   enterClass
    |   enterMethod
    |   enterBlock
    |   varDeclaration
    |   atoms
    ;

bottomup
    :   exitClass
    |   exitMethod
    |   exitBlock
    ;

enterBlock
    :   BLOCK {
            //System.out.println("entering block scope");
            currentScope = new LocalScope(symtab, currentScope);
            System.out.println(currentScope);
        }
    ;

exitBlock
    :   BLOCK {
            //System.out.println("exiting block scope, members: " + currentScope);
            currentScope = currentScope.getEnclosingScope();
        }
    ;

enterMethod
@init {
boolean entry = false;
}
    :   ^((FUNCTION_METHOD_DECL | ENTRY_FUNCTION_DECL {entry = true;})
            (^(MODIFIER_LIST .*))?
            (^(GENERIC_TYPE_PARAM_LIST .*))? 
            type IDENT .*)
        {
            //System.out.println("entering method scope " + $IDENT.text);
            String typeName = $type.text;
            if ($type.text == null) {
                System.out.println("return type of " + $IDENT.text + " has null text, using void");
                typeName = "void";
            }
            //System.out.println("Resolving type " + typeName + " in scope " + currentScope);
            ClassSymbol returnType = currentScope.resolveType(typeName);
            MethodSymbol sym = new MethodSymbol(symtab, $IDENT.text, currentClass, returnType);
            sym.isEntry = entry;
            sym.definition = $enterMethod.start;
            sym.definitionTokenStream = input.getTokenStream();
            currentScope.define($IDENT.text, sym);
            $IDENT.symbol = sym;
            currentScope = sym;
            $IDENT.scope = currentScope;
            System.out.println(currentScope);
        }
    |   ^((CONSTRUCTOR_DECL | ENTRY_CONSTRUCTOR_DECL {entry = true;})
            (^(MODIFIER_LIST .*))?
            (^(GENERIC_TYPE_PARAM_LIST .*))? 
            IDENT .*)
        {
            //System.out.println("entering constructor scope " + $IDENT.text);
            MethodSymbol sym = new MethodSymbol(symtab, $IDENT.text, currentClass, null);
            sym.isEntry = entry;
            sym.isCtor = true;
            sym.definition = $enterMethod.start;
            sym.definitionTokenStream = input.getTokenStream();
            currentScope.define($IDENT.text, sym);
            $IDENT.symbol = sym;
            currentScope = sym;
            $IDENT.scope = currentScope;
            System.out.println(currentScope);
        }
    ;

exitMethod
    :   ^((FUNCTION_METHOD_DECL | ENTRY_FUNCTION_DECL) .*) {
            //System.out.println("exiting method scope: " + currentScope);
            currentScope = currentScope.getEnclosingScope();
        }
    ;

enterClass
    :   ^(TYPE classType IDENT
            (^('extends' parent=type))?
            (^('implements' type+))? .*)
        {
            //System.out.println("Entering class scope");
            ClassSymbol sym = new ClassSymbol(symtab, $IDENT.text,
                    currentScope.resolveType($parent.text), currentScope);
            currentScope.define(sym.name, sym);
            currentClass = sym;
            sym.definition = $IDENT;
            sym.definitionTokenStream = input.getTokenStream();
            $IDENT.symbol = sym;
            currentScope = sym;
            $IDENT.scope = currentScope;
            String classTypeName = $classType.text;
            if (classTypeName.equals("class")) {
            } else if (classTypeName.equals("chare")) {
                currentClass.isChare = true;
            } else if (classTypeName.equals("group")) {
                currentClass.isChare = true;
            } else if (classTypeName.equals("nodegroup")) {
                currentClass.isChare = true;
            } else if (classTypeName.equals("chare_array")) {
                // TODO: test this; might need to use startswith instead of equals
                currentClass.isChare = true;
            } else if (classTypeName.equals("mainchare")) {
                currentClass.isChare = true;
                currentClass.isMainChare = true;
            } else System.out.println("Error: type " + classTypeName + " not recognized.");
            System.out.println(currentScope);
        }
    ;

exitClass
    :   ^(TYPE classType IDENT
            (^('extends' parent=type))?
            (^('implements' type+))? .*)
        {
            //System.out.println("exiting class scope, members: " + currentScope);
            currentScope = currentScope.getEnclosingScope();
        }
    ;

varDeclaration
    :   ^((PRIMITIVE_VAR_DECLARATION | OBJECT_VAR_DECLARATION)
            (^(MODIFIER_LIST .*))? type
            ^(VAR_DECLARATOR_LIST (^(VAR_DECLARATOR ^(IDENT .*) .*)
            {
                //System.out.println("Defining var " + $IDENT.text);
                VariableSymbol sym = new VariableSymbol(symtab, $IDENT.text,
                        currentScope.resolveType($type.text));
                sym.definition = $IDENT;
                sym.definitionTokenStream = input.getTokenStream();
                $IDENT.symbol = sym;
                $IDENT.scope = currentScope;
                currentScope.define($IDENT.text, sym);
            }
            )+))
    |   ^(FORMAL_PARAM_STD_DECL (^(MODIFIER_LIST .*))? type ^(IDENT .*))
        {
            //System.out.println("Defining argument var " + $IDENT.text);
            VariableSymbol sym = new VariableSymbol(symtab, $IDENT.text,
                    currentScope.resolveType($type.text));
            sym.definition = $IDENT;
            sym.definitionTokenStream = input.getTokenStream();
            $IDENT.symbol = sym;
            $IDENT.scope = currentScope;
            currentScope.define($IDENT.text, sym);
        }
    ;

atoms
@init {CharjAST t = (CharjAST)input.LT(1);}
    :  {t.hasAncestor(EXPR)}? (IDENT|THIS|SUPER) {
            t.scope = currentScope;
            System.out.println(currentScope);
       }
    ;

type
@init { if (currentScope == null) System.out.println("*****ERROR: null type scope"); }
    :   VOID { $VOID.scope = currentScope; }
    |   ^(SIMPLE_TYPE .*) { $SIMPLE_TYPE.scope = currentScope; }
    |   ^(OBJECT_TYPE ^(QUALIFIED_TYPE_IDENT (^(IDENT .*))+) .*)    { $OBJECT_TYPE.scope = currentScope; }
    |   ^(REFERENCE_TYPE ^(QUALIFIED_TYPE_IDENT (^(IDENT .*))+) .*) { $REFERENCE_TYPE.scope = currentScope; }
    |   ^(PROXY_TYPE ^(QUALIFIED_TYPE_IDENT (^(IDENT .*))+) .*)     { $PROXY_TYPE.scope = currentScope; }
    |   ^(POINTER_TYPE ^(QUALIFIED_TYPE_IDENT (^(IDENT .*))+) .*)   { $POINTER_TYPE.scope = currentScope; }
    ;

classType
    :   CLASS
    |   CHARE
    |   GROUP
    |   NODEGROUP
    |   MAINCHARE
    |   ^(CHARE_ARRAY ARRAY_DIMENSION)
    ;
