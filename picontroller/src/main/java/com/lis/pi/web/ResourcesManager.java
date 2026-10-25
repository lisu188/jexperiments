package com.lis.pi.web;

import java.io.InputStream;
import java.net.URL;
import java.util.Map;

import static java.util.Map.entry;

public class ResourcesManager {
    private static ResourcesManager instance = new ResourcesManager();
    private static final String ROOT = "/com/lis/pi/web/";
    // Requests select public assets, never arbitrary files in the application's classpath.
    private static final Map<String, String> PUBLIC_ASSETS = Map.ofEntries(
            entry("", ROOT + "index.html"),
            entry("index.html", ROOT + "index.html"),
            entry("js/servlet.js", ROOT + "js/servlet.js"),
            entry("js/dashboard-controls.js", ROOT + "js/dashboard-controls.js"),
            entry("js/bootstrap.js", ROOT + "js/bootstrap.js"),
            entry("js/bootstrap.min.js", ROOT + "js/bootstrap.min.js"),
            entry("css/bootstrap.css", ROOT + "css/bootstrap.css"),
            entry("css/bootstrap.min.css", ROOT + "css/bootstrap.min.css"),
            entry("css/bootstrap.css.map", ROOT + "css/bootstrap.css.map"),
            entry("css/bootstrap-theme.css", ROOT + "css/bootstrap-theme.css"),
            entry("css/bootstrap-theme.min.css", ROOT + "css/bootstrap-theme.min.css"),
            entry("css/bootstrap-theme.css.map", ROOT + "css/bootstrap-theme.css.map"),
            entry("fonts/glyphicons-halflings-regular.eot", ROOT + "fonts/glyphicons-halflings-regular.eot"),
            entry("fonts/glyphicons-halflings-regular.svg", ROOT + "fonts/glyphicons-halflings-regular.svg"),
            entry("fonts/glyphicons-halflings-regular.ttf", ROOT + "fonts/glyphicons-halflings-regular.ttf"),
            entry("fonts/glyphicons-halflings-regular.woff", ROOT + "fonts/glyphicons-halflings-regular.woff"));

    private ResourcesManager() {
    }

    public static ResourcesManager getInstance() {
        return instance;
    }

    public URL getWebResource(String path) {
        String resource = path == null ? null : PUBLIC_ASSETS.get(path);
        return resource == null ? null : ResourcesManager.class.getResource(resource);
    }

    public InputStream getWebResourceAsStream(String path) {
        String resource = path == null ? null : PUBLIC_ASSETS.get(path);
        return resource == null ? null : ResourcesManager.class.getResourceAsStream(resource);
    }

}
