
package charj.translator;

public class ProxyType extends Symbol implements Type {
    public Type baseType;

    public ProxyType(SymbolTable symtab, Type _baseType) {
        super(symtab, _baseType.getTypeName() + "@", null);
        baseType = _baseType;
    }

    public String getTypeName() {
        return baseType.getTypeName() + "@";
    }

    public String getTranslatedTypeName() {
        return "CProxy_" + baseType.getTypeName();
    }
}
