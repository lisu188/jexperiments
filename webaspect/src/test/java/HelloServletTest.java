import org.junit.jupiter.api.Test;

import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;
import java.io.File;
import java.lang.reflect.Proxy;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

class HelloServletTest {
    @Test
    void requestPreservesSessionBehaviorWithoutStartingAProcessExitTimer() throws Exception {
        var classpath = String.join(File.pathSeparator,
                location(HelloServlet.class), location(HelloServletTest.class), location(HttpServlet.class));
        var java = Path.of(System.getProperty("java.home"), "bin", "java").toString();
        var process = new ProcessBuilder(java, "-cp", classpath, ServletProbe.class.getName())
                .redirectErrorStream(true)
                .start();
        try {
            assertTrue(process.waitFor(10, TimeUnit.SECONDS),
                    "Servlet initialization/request must not leave a process-exit timer running");
            var output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            assertEquals(0, process.exitValue(), output);
            assertTrue(output.contains("session-preserved"), output);
        } finally {
            if (process.isAlive()) {
                process.destroyForcibly();
                process.waitFor(5, TimeUnit.SECONDS);
            }
        }
    }

    private static String location(Class<?> type) throws Exception {
        return Path.of(type.getProtectionDomain().getCodeSource().getLocation().toURI()).toString();
    }

    public static class ServletProbe {
        public static void main(String[] args) throws Exception {
            var attributes = new HashMap<String, Object>();
            var loader = HelloServlet.class.getClassLoader();
            var session = (HttpSession) Proxy.newProxyInstance(loader, new Class<?>[]{HttpSession.class},
                    (proxy, method, arguments) -> {
                        if (method.getName().equals("setAttribute")) {
                            attributes.put((String) arguments[0], arguments[1]);
                        }
                        return null;
                    });
            var request = (HttpServletRequest) Proxy.newProxyInstance(loader,
                    new Class<?>[]{HttpServletRequest.class},
                    (proxy, method, arguments) -> method.getName().equals("getSession") ? session : null);
            var response = (HttpServletResponse) Proxy.newProxyInstance(loader,
                    new Class<?>[]{HttpServletResponse.class}, (proxy, method, arguments) -> null);
            new HelloServlet().doGet(request, response);
            if (!"UWAGA".equals(attributes.get("halo janusz"))) {
                throw new AssertionError("GET no longer writes the expected session attribute");
            }
            System.out.println("session-preserved");
        }
    }
}
