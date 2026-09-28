package experiments.kotlin.propertyrefs;

public final class JavaPropertyCaller {
    private JavaPropertyCaller() {
    }

    public static String call(PropertyBox box) {
        return PropertyBox.CONST_NAME
                + "|" + PropertyBox.exposedField
                + "|" + PropertyBox.staticAccessor()
                + "|" + box.getMutable()
                + "|" + box.directJvmField
                + "|" + PropertySingleton.objectField
                + "|" + PropertySingleton.objectCall();
    }
}
