package com.lis.pi;

import org.apache.catalina.Context;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.Proxy;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import org.apache.catalina.startup.Tomcat;

import static org.junit.jupiter.api.Assertions.*;

class TomcatManagerTest {
    @Test
    void dashboardDoesNotRegisterRemoteShutdownHandler() {
        var tomcat = new TomcatManager().getTomcat();
        var context = (Context) tomcat.getHost().findChild("");

        assertNotNull(context);
        assertNull(context.findServletMapping("/kill"));
        assertNull(context.findChild("Kill"));
        assertEquals("Default", context.findServletMapping("/"));
    }

    @Test
    void configuredConnectorStartsAnHttpListenerWithoutPiHardware(@TempDir Path directory) throws Exception {
        var tomcat = new TomcatManager().getTomcat();
        var connectors = tomcat.getService().findConnectors();
        assertEquals(1, connectors.length, "Configuration must create the HTTP connector before start");
        assertEquals(8080, connectors[0].getPort());
        connectors[0].setPort(0);
        connectors[0].setProperty("address", "127.0.0.1");
        tomcat.setBaseDir(directory.toString());
        // Replace only the hardware-bound context; retain the manager's actual
        // connector and verify its network startup with a minimal HTTP route.
        tomcat.getHost().removeChild(tomcat.getHost().findChild(""));
        var context = tomcat.addContext("", directory.toString());
        Tomcat.addServlet(context, "probe", new HttpServlet() {
            @Override
            protected void doGet(HttpServletRequest request, HttpServletResponse response) throws IOException {
                response.getWriter().write("listener-ready");
            }
        });
        context.addServletMappingDecoded("/probe", "probe");
        try {
            tomcat.start();
            var url = new URL("http://127.0.0.1:" + connectors[0].getLocalPort() + "/probe");
            var connection = (HttpURLConnection) url.openConnection(Proxy.NO_PROXY);
            connection.setConnectTimeout(5000);
            connection.setReadTimeout(5000);
            try {
                assertEquals(200, connection.getResponseCode());
                try (var input = connection.getInputStream()) {
                    assertEquals("listener-ready", new String(input.readAllBytes(), StandardCharsets.UTF_8));
                }
            } finally {
                connection.disconnect();
            }
        } finally {
            tomcat.stop();
            tomcat.destroy();
        }
    }
}
