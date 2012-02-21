
package charj.translator;

public class ExternalSymbol extends ClassSymbol implements Scope, Type {
    
    public ExternalSymbol(SymbolTable symtab, String name) {
        super(symtab, name);
    }
}

