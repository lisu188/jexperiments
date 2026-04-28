package experiments.kotlin.datasynthetic;

import java.lang.reflect.Method;

public final class JavaDataCaller {
    private JavaDataCaller() {
    }

    public static String callDataApi() {
        try {
            BuildCoordinate coordinate = new BuildCoordinate("experiments", "data", 1);
            BuildCoordinate copy = coordinate.copy("experiments", "data", 2);
            Method copyDefault = BuildCoordinate.class.getDeclaredMethod(
                    "copy$default",
                    BuildCoordinate.class,
                    String.class,
                    String.class,
                    int.class,
                    int.class,
                    Object.class);
            Object defaulted = copyDefault.invoke(null, coordinate, null, null, 3, 0b001 | 0b010, null);
            return coordinate.component1() + "|" + copy.gav() + "|" + defaulted;
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException(e);
        }
    }
}
