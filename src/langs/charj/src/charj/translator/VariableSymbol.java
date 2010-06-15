
package charj.translator;

/** Represents a variable definition (name,type) in symbol table (a scope 
 *  thereof)
 */
public class VariableSymbol extends Symbol {
    public boolean isStatic = false;
    public boolean isConst = false;

    public VariableSymbol(
            SymbolTable symtab,
            String name,
            Type type) {
        super(symtab, name, type);
    }

    public String toString() {
        StringBuffer buf = new StringBuffer();
        if ( scope!=null ) {
            buf.append(scope.getScopeName());
            buf.append(".");
        }
        buf.append(name);
        return buf.toString();
    }
}
