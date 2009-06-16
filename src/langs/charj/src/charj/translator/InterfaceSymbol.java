
package charj.translator;

import java.util.*;

public class InterfaceSymbol extends ClassSymbol {
    public List<String> superClasses = new ArrayList<String>();

    public InterfaceSymbol(
            SymbolTable symtab, 
            String name, 
            ClassSymbol superClass, 
            Scope scope) {
        super(symtab, name, superClass, scope);
    }
}
