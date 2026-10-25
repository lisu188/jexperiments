package com.lis.pi;

import org.apache.catalina.Context;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class TomcatManagerTest {
    @Test
    void dashboardDoesNotRegisterRemoteShutdownHandler() {
        var tomcat = TomcatManager.getInstance().getTomcat();
        var context = (Context) tomcat.getHost().findChild("");

        assertNotNull(context);
        assertNull(context.findServletMapping("/kill"));
        assertNull(context.findChild("Kill"));
        assertEquals("Default", context.findServletMapping("/"));
    }
}
