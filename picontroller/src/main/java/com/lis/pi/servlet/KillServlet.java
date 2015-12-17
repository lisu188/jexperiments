package com.lis.pi.servlet;

import com.lis.pi.TomcatManager;
import org.apache.logging.log4j.LogManager;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

public class KillServlet extends HttpServlet {

    @Override
    protected void doGet(HttpServletRequest req, HttpServletResponse resp)
            throws ServletException, IOException {
        new Thread() {
            @Override
            public void run() {
                try {
                    sleep(500);
                    TomcatManager.getInstance().getTomcat().stop();
                } catch (Exception e) {
                    LogManager.getLogger(getClass()).error("Error", e);
                }
            }
        }.start();
    }
}