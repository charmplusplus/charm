package charj.translator;

public class ProxySectionType extends Symbol implements Type {
    public Type baseType;

    public ProxySectionType(SymbolTable symtab, Type _baseType) {
        super(symtab, _baseType.getTypeName(), null);
        baseType = _baseType;
    }

    public String getTypeName() {
        return baseType.getTypeName();
    }

	public String getTranslatedTypeName()
	{
        return "CProxySection_" + baseType.getTypeName();
	}
}
