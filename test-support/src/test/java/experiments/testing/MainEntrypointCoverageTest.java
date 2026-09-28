package experiments.testing;

import org.junit.jupiter.api.Test;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.Comparator;

import static org.junit.jupiter.api.Assertions.assertNotNull;

class MainEntrypointCoverageTest {
    @Test
    void configuredEntrypointsRun() throws Throwable {
        String configured = System.getProperty("coverage.smokeMains", "").trim();
        if (configured.isEmpty()) {
            return;
        }

        for (String className : configured.split(",")) {
            invokeMain(className.trim());
        }
    }

    private static void invokeMain(String className) throws Throwable {
        Class<?> type = Class.forName(className);
        Method main = Arrays.stream(type.getDeclaredMethods())
                .filter(method -> method.getName().equals("main"))
                .filter(method -> Modifier.isStatic(method.getModifiers()))
                .filter(MainEntrypointCoverageTest::supportedMain)
                .min(Comparator.comparingInt(Method::getParameterCount))
                .orElse(null);

        assertNotNull(main, () -> "No supported main entrypoint on " + className);
        main.setAccessible(true);

        try {
            if (main.getParameterCount() == 0) {
                main.invoke(null);
            } else {
                main.invoke(null, (Object) new String[0]);
            }
        } catch (InvocationTargetException failure) {
            throw failure.getCause();
        }
    }

    private static boolean supportedMain(Method method) {
        if (method.getParameterCount() == 0) {
            return true;
        }
        return method.getParameterCount() == 1 && method.getParameterTypes()[0] == String[].class;
    }
}
