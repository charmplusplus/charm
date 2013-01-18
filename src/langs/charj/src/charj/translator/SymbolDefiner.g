
tree grammar SymbolDefiner;

options {
    tokenVocab = Charj;
    ASTLabelType = CharjAST;
    filter = true;
}

@header {
package charj.translator;
import java.util.HashSet;

}

@members {
    SymbolTable symtab;
    Scope currentScope = null;
    ClassSymbol currentClass = null;
    MethodSymbol currentMethod = null;
    AstModifier astmod = new AstModifier();

    public SymbolDefiner(TreeNodeStream input, SymbolTable symtab) {
        this(input);
        this.symtab = symtab;
        this.currentScope = symtab.getDefaultPkg();
    }
}


topdown
    :   enterPackage
    |   externDeclaration
    |   enterClass
    |   enterMethod
    |   enterBlock
    |   varDeclaration
    |   type
    |   atoms
    ;

bottomup
    :   exitClass
    |   exitMethod
    |   exitBlock
    ;

enterPackage
@init {
    List<String> names = null;
    String packageName = "";
}
    :   ^(PACKAGE ((ids+=IDENT) { packageName += "." + $IDENT.text; })+)
        {
            packageName = packageName.substring(1);
            PackageScope ps = symtab.resolvePackage(packageName);
            if (ps == null) {
                ps = symtab.definePackage(packageName);
                symtab.addScope(ps);
            }
            currentScope = ps;
            assert currentScope != null;
        }
    ;

externDeclaration
    :   ^(EXTERN IDENT) {
            ExternalSymbol sym = new ExternalSymbol(symtab, $IDENT.text);
            currentScope.define(sym.name, sym);
        }
    ;

enterBlock
    :   BLOCK {
            $BLOCK.scope = new LocalScope(symtab, currentScope);
            $BLOCK.def = (LocalScope)$BLOCK.scope;
            currentScope = $BLOCK.scope;
            assert currentScope != null;
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
            assert currentScope != null;
            //System.out.println("entering method scope " + $IDENT.text + " " + currentScope);
            boolean isTraced = false;
            boolean sdagEntry = false;
            if ($MODIFIER_LIST != null) {
                CharjAST charj_mod = $MODIFIER_LIST.getChildOfType(CharjParser.CHARJ_MODIFIER_LIST);
                if (charj_mod != null) {
                    charj_mod = charj_mod.getChildOfType(CharjParser.TRACED);
                    isTraced = (charj_mod != null);
                    if (isTraced) System.out.println("method " + $IDENT.text + " is traced");
                }
                charj_mod = $MODIFIER_LIST.getChildOfType(CharjParser.CHARJ_MODIFIER_LIST);
                if (charj_mod != null) {
                    charj_mod = charj_mod.getChildOfType(CharjParser.SDAGENTRY);
                    sdagEntry = (charj_mod != null);
                }
            }
            Type returnType = $type.namedType;
            /*System.out.println("Resolving return type in scope " +
                    currentScope + "->" + returnType);*/
            MethodSymbol sym = new MethodSymbol(symtab, $IDENT.text, currentClass, returnType);
            sym.isEntry = entry;
            sym.isTraced = isTraced;
            sym.definition = $enterMethod.start;
            sym.hasSDAG = sdagEntry;
            sym.definitionTokenStream = input.getTokenStream();
            currentScope.define($IDENT.text, sym);
            currentScope = sym;
            currentMethod = sym;
            $IDENT.def = sym;
            $IDENT.symbolType = sym.type;
            $IDENT.scope = currentScope;
            //System.out.println(currentScope);
        }
    |   ^((CONSTRUCTOR_DECL
          | ENTRY_CONSTRUCTOR_DECL {
                entry = true;
            })
            (^(MODIFIER_LIST .*))?
            (^(GENERIC_TYPE_PARAM_LIST .*))? 
            IDENT .*)
        {
            assert currentScope != null;
            //System.out.println("entering constructor scope " + $IDENT.text);
            boolean isTraced = false;
            CharjAST charj_mod = $MODIFIER_LIST.getChildOfType(CharjParser.CHARJ_MODIFIER_LIST);
            if (charj_mod != null) {
                charj_mod = charj_mod.getChildOfType(CharjParser.TRACED);
                isTraced = (charj_mod != null);
                if (isTraced) System.out.println("method " + $IDENT.text + " is traced");
            }
            MethodSymbol sym = new MethodSymbol(symtab, $IDENT.text, currentClass, currentClass);
            sym.isEntry = entry;
            sym.isCtor = true;
            sym.isTraced = isTraced;
            sym.definition = $enterMethod.start;
            sym.definitionTokenStream = input.getTokenStream();
            currentScope.define($IDENT.text, sym);
            currentScope = sym;
            currentMethod = sym;
            $IDENT.def = sym;
            $IDENT.symbolType = sym.type;
            $IDENT.scope = currentScope;
            //System.out.println(currentScope);
        }
    ;

exitMethod
    :   ^((FUNCTION_METHOD_DECL | ENTRY_FUNCTION_DECL | CONSTRUCTOR_DECL | ENTRY_CONSTRUCTOR_DECL) .*) {
            //System.out.println("method " + currentScope);
            assert currentScope != null;
            currentScope = currentScope.getEnclosingScope();
            currentMethod = null;
        }
    ;


enterClass
    :   ^(TYPE classType IDENT
            (^('extends' parent=type))?
            (^('implements' type+))?
            (^((FUNCTION_METHOD_DECL | ENTRY_FUNCTION_DECL | DIVCON_METHOD_DECL | PRIMITIVE_VAR_DECLARATION |
                OBJECT_VAR_DECLARATION | CONSTRUCTOR_DECL | ENTRY_CONSTRUCTOR_DECL) .*))*)
        {
            //System.out.println("Defined class " + $IDENT.text);
            ClassSymbol sym = new ClassSymbol(symtab, $IDENT.text,
                    (ClassSymbol)currentScope.
                      resolveType(TypeName.createTypeName($parent.text)),
                                  currentScope);
            currentScope.define(sym.name, sym);
            currentClass = sym;
            sym.definition = $IDENT;
            sym.definitionTokenStream = input.getTokenStream();
            currentScope = sym;
            $IDENT.def = sym;
            $IDENT.scope = currentScope;
            $IDENT.symbolType = sym;
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
                // TODO: should "isChare" be set to true?
                currentClass.isChare = true;
                currentClass.isChareArray = true;
            } else if (classTypeName.equals("mainchare")) {
                currentClass.isChare = true;
                currentClass.isMainChare = true;
            } else System.out.println("Error: type " + classTypeName + " not recognized.");
        }
    ;

exitClass
    :   ^(TYPE classType IDENT
            (^('extends' parent=type))?
            (^('implements' type+))?
            (^((FUNCTION_METHOD_DECL | ENTRY_FUNCTION_DECL |  DIVCON_METHOD_DECL | PRIMITIVE_VAR_DECLARATION |
                OBJECT_VAR_DECLARATION | CONSTRUCTOR_DECL | ENTRY_CONSTRUCTOR_DECL) .*))*)
        {
            //System.out.println("class " + currentScope);
            currentScope = currentScope.getEnclosingScope();
        }
    ;

varDeclaration
    :   ^((PRIMITIVE_VAR_DECLARATION | OBJECT_VAR_DECLARATION)
            (^(MODIFIER_LIST .*))? type
            ^(VAR_DECLARATOR_LIST (^(VAR_DECLARATOR ^(IDENT .*) .*)
            {
                Type varType = $type.namedType;
                /*System.out.println("Defining var " + $IDENT.text + " with type " +
                    varType);*/
                VariableSymbol sym = new VariableSymbol(symtab, $IDENT.text, varType);
                sym.definition = $IDENT;
                sym.definitionTokenStream = input.getTokenStream();
                if (currentScope instanceof PackageScope) {
                    sym.isReadOnly = true;
                    //System.out.println("Marking " + $IDENT.text + " as readonly");
                }
                $IDENT.def = sym;
                $IDENT.scope = currentScope;
                $IDENT.symbolType = varType;
                currentScope.define($IDENT.text, sym);
            }
            )+))
    |   ^(FORMAL_PARAM_STD_DECL (^(MODIFIER_LIST .*))? type ^(IDENT .*))
        {
            Type varType = $type.namedType;
            //System.out.println("Defining argument var " + $IDENT.text + " with type " + varType);
            VariableSymbol sym = new VariableSymbol(symtab, $IDENT.text, varType);
            sym.definition = $IDENT;
            sym.definitionTokenStream = input.getTokenStream();
            $IDENT.def = sym;
            $IDENT.scope = currentScope;
            $IDENT.symbolType = varType;
            currentScope.define($IDENT.text, sym);
        }
    ;


type returns [Type namedType]
@init {
    ArrayList<TypeName> typeName = new ArrayList<TypeName>();
    if (currentScope == null) System.out.println("*****ERROR: null type scope");
    assert currentScope != null;
}
@after {
    // TODO: Special case for Arrays, change this?
    String name = typeName.get(0).name;
    HashSet<String> s = new HashSet();
    s.add("Array"); s.add("Matrix"); s.add("Vector");
    if (typeName.size() > 0 && s.contains(name) &&
            $namedType == null) {

        int numDims = 1;
        ClassSymbol cs = new ClassSymbol(symtab, name);
        cs.templateArgs = typeName.get(0).parameters;

        if (name.equals("Array") &&
            cs.templateArgs != null &&
            cs.templateArgs.size() > 1) {
            if (cs.templateArgs.get(1) instanceof LiteralType) {
                numDims = Integer.valueOf(
                    ((LiteralType)cs.templateArgs.get(1)).literal);
            }
        } else if (name.equals("Vector"))
            numDims = 1;
        else if (name.equals("Matrix"))
            numDims = 2;

        $namedType = new PointerType(symtab, cs);
    }
}
    :   VOID {
            $VOID.scope = currentScope;
            typeName.add(new TypeName("void"));
            $namedType = currentScope.resolveType(typeName);
        }
    |   ^(SIMPLE_TYPE t=. .*) {
            $SIMPLE_TYPE.scope = currentScope;
            typeName.add(new TypeName($t.toString()));
            $namedType = currentScope.resolveType(typeName);
        }
    |   ^(OBJECT_TYPE ^(QUALIFIED_TYPE_IDENT (^(i1=IDENT {typeName.add(new TypeName($IDENT.text));} .*))+) .*)
            {
                $OBJECT_TYPE.scope = currentScope;
                $namedType = currentScope.resolveType(typeName);
            }
    |   ^(REFERENCE_TYPE ^(QUALIFIED_TYPE_IDENT (^(IDENT {typeName.add(new TypeName($IDENT.text));} .*))+) .*)
            {
                $REFERENCE_TYPE.scope = currentScope;
                Type base = currentScope.resolveType(typeName);
                $namedType = base == null ? null : new PointerType(symtab, base);
            }
    |   ^(PROXY_TYPE ^(QUALIFIED_TYPE_IDENT (^(IDENT {typeName.add(new TypeName($IDENT.text));} .*))+) .*)
            {
                $PROXY_TYPE.scope = currentScope;
                Type base = currentScope.resolveType(typeName);
                $namedType = base == null ? null : new ProxyType(symtab, base);
            }
    |   ^(POINTER_TYPE ^(QUALIFIED_TYPE_IDENT (^(i1=IDENT {typeName.add(new TypeName($i1.text));} .*))+) .*)
            {
                $POINTER_TYPE.scope = currentScope;
                Type base = currentScope.resolveType(typeName);
                $namedType = base == null ? null : new PointerType(symtab, base);
            }
    |   ^(MESSAGE_TYPE ^(QUALIFIED_TYPE_IDENT (^(i1=IDENT {typeName.add(new TypeName($i1.text));} .*))+) .*)
            {
                $MESSAGE_TYPE.scope = currentScope;
                Type base = currentScope.resolveType(typeName);
                $namedType = base == null ? null : new MessageType(symtab, base);
            }

    |   ^(ARRAY_SECTION_TYPE ^(QUALIFIED_TYPE_IDENT (^(i1=IDENT {typeName.add(new TypeName($i1.text));} .*))+) .*)
			{
                $ARRAY_SECTION_TYPE.scope = currentScope;
                Type base = currentScope.resolveType(typeName);
                $namedType = base == null ? null : new ProxySectionType(symtab, base);
            }
    ;

literal returns [String lit]
@init {
$lit = $start.getText().toString();
}
    :   HEX_LITERAL
    |   OCTAL_LITERAL
    |   DECIMAL_LITERAL
    |   FLOATING_POINT_LITERAL
    |   CHARACTER_LITERAL
    |   STRING_LITERAL
    |   TRUE
    |   FALSE
    |   NULL
    ;

classType
    :   CLASS
    |   CHARE
    |   GROUP
    |   NODEGROUP
    |   MAINCHARE
    |   ^(CHARE_ARRAY ARRAY_DIMENSION)
    ;

atoms
@init {CharjAST t = (CharjAST)input.LT(1);}
    :  (IDENT|THIS|SUPER) {
            assert currentScope != null;
            t.scope = currentScope;
       }
    |  ^(WHEN sdagTrigger+ ^(BLOCK .*)) {
            assert currentMethod != null;
            currentMethod.hasSDAG = true;
       }
    |  OVERLAP {
            assert currentMethod != null;
            currentMethod.hasSDAG = true;
       }
    ;

sdagTrigger
    :  IDENT .* ^(FORMAL_PARAM_LIST .*) {
         ArrayList<TypeName> typeName = new ArrayList<TypeName>();
         typeName.add(new TypeName("void"));
         Type returnType = currentScope.resolveType(typeName);
         MethodSymbol sym = new MethodSymbol(symtab, $IDENT.text, currentClass, returnType);
         sym.sdagFPL = $FORMAL_PARAM_LIST;
         sym.isEntry = true;
         // @todo fix this??
         sym.isTraced = false;
         //sym.definition = $IDENT.start;
         sym.hasSDAG = true;
         sym.definitionTokenStream = input.getTokenStream();
         currentScope.define($IDENT.text, sym);
         //currentScope = sym;
         //currentMethod = sym;
         $IDENT.def = sym;
         $IDENT.symbolType = sym.type;
         $IDENT.scope = currentScope;
         currentClass.sdagMethods.put($IDENT.text, sym);
         currentScope.define($IDENT.text, sym);
      }
    ;
