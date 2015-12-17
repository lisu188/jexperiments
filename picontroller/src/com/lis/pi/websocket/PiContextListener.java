package com.lis.pi.websocket;

import javax.servlet.ServletContext;
import javax.servlet.ServletContextEvent;
import javax.websocket.DeploymentException;
import javax.websocket.server.ServerContainer;

import org.apache.logging.log4j.LogManager;
import org.apache.tomcat.websocket.server.WsContextListener;

public class PiContextListener extends WsContextListener {
	private final ServletContext context;

	public PiContextListener(ServletContext context) {
		super();
		this.context = context;
	}

	@Override
	public void contextDestroyed(ServletContextEvent sce) {
		ServletContextEvent event = new ServletContextEvent(context);
		super.contextDestroyed(event);

	}

	@Override
	public void contextInitialized(ServletContextEvent sce) {
		try {
			ServletContextEvent event = new ServletContextEvent(context);
			super.contextInitialized(event);
			final ServerContainer serverContainer = (ServerContainer) sce
					.getServletContext().getAttribute(
							"javax.websocket.server.ServerContainer");
			serverContainer.addEndpoint(new InfoEndpointConfig());
		} catch (DeploymentException e) {
			LogManager.getLogger(getClass()).error("Error", e);
		}

	}
}
