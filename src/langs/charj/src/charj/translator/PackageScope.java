
package charj.translator;

import java.util.HashMap;
import java.util.Map;
import java.util.List;
import java.util.ArrayList;

public class PackageScope extends SymbolWithScope {

    /** List of packages and classes in this package */
    Map<String, Symbol> members = new HashMap(); // union of types and subpackages
    Scope enclosingScope;

    public PackageScope(
            SymbolTable symtab, 
            String name, 
            Scope enclosingScope) {
        super(symtab, name);
        this.enclosingScope = enclosingScope;
    }

    public Scope getEnclosingScope() {
        return enclosingScope;
    }

    /** See if type is already defined in this package.  If not, look
     *  for type on the disk in same package.  For example, first time
     *  charj.lang.Chare fails to resolve.  Load from disk and put Chare
     *  in package lang which is in package charj.  Next time, Chare will
     *  be found.
     */
    public ClassSymbol resolveType(List<TypeName> type) {
        if (type == null) return null;

        String typeStr = TypeName.typeToString(type);

        if (debug()) { 
            System.out.println(" PackageScope.resolveType(" + typeStr + 
                                "): examine " + toString());
        }

        ClassSymbol cs = symtab.lookupPrimitive(type);
        if (cs != null) return cs;

        if (type.size() == 1) return (ClassSymbol)members.get(type.get(0).name);
        PackageScope innerPackage = (PackageScope)members.get(type.get(0).name);
        if (innerPackage == null) {
            if (debug()) System.out.println("Package lookup for " +
                                            type.get(0) + "failed.\n");
            return null;
        }

        return innerPackage.resolveType(TypeName.createTypeName(type.get(1).name));
    }

    public String getFullyQualifiedName() {
        if ( name.equals(SymbolTable.DEFAULT_PACKAGE_NAME) ) {
            return null;
        }
        return super.getFullyQualifiedName();
    }

    public Map<String, Symbol> getMembers() {
        return members;
    }

    public String toString() {
        return "PackageScope[" + name + "]: " + members.keySet();
    }
}
