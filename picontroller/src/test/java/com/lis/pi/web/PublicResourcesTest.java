package com.lis.pi.web;

import org.junit.jupiter.api.Test;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import static org.junit.jupiter.api.Assertions.*;

class PublicResourcesTest {
    @Test void servesTheDashboardAndItsPublicAssets() throws Exception {
        ResourcesManager resources = ResourcesManager.getInstance();
        for (String path : new String[]{"", "index.html", "js/servlet.js", "js/dashboard-controls.js", "css/bootstrap.min.css", "fonts/glyphicons-halflings-regular.woff"}) {
            assertNotNull(resources.getWebResource(path), path);
            try (InputStream input = resources.getWebResourceAsStream(path)) {
                assertNotNull(input, path);
                assertTrue(input.readAllBytes().length > 0, path);
            }
        }
        try (InputStream input = resources.getWebResourceAsStream("")) {
            assertTrue(new String(input.readAllBytes(), StandardCharsets.UTF_8).contains("<html"));
        }
    }
    @Test void cannotReadClasspathCodeOrTraverseOutsidePublicAssets() {
        ResourcesManager resources = ResourcesManager.getInstance();
        for (String path : new String[]{"../servlet/DefaultServlet.class", "ResourcesManager.class", "/hrab/WeatherQuery.class", "js/../ResourcesManager.class", "%2e%2e/secret", "js\\servlet.js", "unknown.js", "index.html?other", "./index.html"}) {
            assertNull(resources.getWebResource(path), path);
            assertNull(resources.getWebResourceAsStream(path), path);
        }
        for (String path : new String[]{"js/bootstrap.js", "js/bootstrap.min.js"}) {
            assertNull(resources.getWebResource(path), path);
            assertNull(resources.getWebResourceAsStream(path), path);
        }
        assertNull(resources.getWebResource(null));
        assertNull(resources.getWebResourceAsStream(null));
    }
}
