
package charj.translator;

import charj.translator.CharjAST;
import org.antlr.runtime.TokenStream;

public class Symbol {
    /** All symbols at least have a name */
    public String name;

    /** Classes, methods, variables, and closures have types */
    public Type type;

    /** All symbols know what scope contains them. */
    public Scope scope;

    /** All symbols know which symbol table they are apart of */
    public SymbolTable symtab;

    /** Often we want to know where in tree this symbol was defined. */
    public CharjAST definition;

    /** To print definition, we need to know where tokens live */
    public TokenStream definitionTokenStream;

    public Symbol(SymbolTable _symtab) {
        symtab = _symtab;
    }

    public Symbol(
            SymbolTable _symtab, 
            String _name, 
            Type _type) {
        this(_symtab);
        name = _name;
        type = _type;
    }

    public boolean debug() {
        return symtab.translator.debug();
    }

    public String toString() {
        String info = "";
        if (name != null)
            info += name + ", ";
        if (type != null)
            info += type + ", ";
        if (scope != null)
            info += scope;
        return "Symbol(" + info + ")";
    }


}
