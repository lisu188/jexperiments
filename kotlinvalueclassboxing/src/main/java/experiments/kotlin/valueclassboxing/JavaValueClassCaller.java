package experiments.kotlin.valueclassboxing;

import java.lang.reflect.Method;

public final class JavaValueClassCaller {
    private JavaValueClassCaller() {
    }

    public static String reflectCustomerId(String raw) {
        try {
            Class<?> idClass = Class.forName("experiments.kotlin.valueclassboxing.CustomerId");
            Method box = idClass.getDeclaredMethod("box-impl", String.class);
            Method unbox = idClass.getDeclaredMethod("unbox-impl");
            box.setAccessible(true);
            unbox.setAccessible(true);
            Object boxed = box.invoke(null, raw);
            Object unboxed = unbox.invoke(boxed);
            return boxed.getClass().getSimpleName() + "|raw=" + unboxed;
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException(e);
        }
    }
}
