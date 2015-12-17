package com.lis.pi;

import com.lis.pi.servlet.DefaultServlet;
import com.lis.pi.servlet.KillServlet;
import com.lis.pi.websocket.PiContextListener;
import org.apache.catalina.Context;
import org.apache.catalina.LifecycleEvent;
import org.apache.catalina.LifecycleState;
import org.apache.catalina.core.StandardContext;
import org.apache.catalina.startup.Tomcat;
import org.apache.catalina.startup.Tomcat.FixContextListener;

import java.io.File;

public class TomcatManager {
    private static TomcatManager instance = new TomcatManager();
    private final Tomcat tomcat;

    private TomcatManager() {
        this.tomcat = new Tomcat();
        tomcat.setPort(8080);
        File base = new File(System.getProperty("java.io.tmpdir"));
        final Context rootCtx = new StandardContext();
        rootCtx.setName("");
        rootCtx.setPath("");
        rootCtx.setDocBase(base.getAbsolutePath());
        rootCtx.addLifecycleListener(new FixContextListener() {
            private volatile boolean added = false;

            @Override
            public void lifecycleEvent(LifecycleEvent event) {
                super.lifecycleEvent(event);
                if (rootCtx.getState().equals(LifecycleState.STARTING_PREP)) {
                    if (!added) {
                        rootCtx.getServletContext().addListener(
                                new PiContextListener(rootCtx
                                        .getServletContext()));
                        added = true;
                    }
                }
            }

        });

        tomcat.getHost().addChild(rootCtx);

        Tomcat.addServlet(rootCtx, "Kill", new KillServlet());
        rootCtx.addServletMapping("/kill", "Kill");
        Tomcat.addServlet(rootCtx, "Default", new DefaultServlet());
        rootCtx.addServletMapping("/", "Default");
    }

    public static TomcatManager getInstance() {
        return instance;
    }

    public Tomcat getTomcat() {
        return tomcat;
    }
}
