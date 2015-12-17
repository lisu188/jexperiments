package com.lis.pi.websocket;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

import javax.websocket.CloseReason;
import javax.websocket.Endpoint;
import javax.websocket.EndpointConfig;
import javax.websocket.Session;

import com.lis.pi.monitor.Monitor;

public class InfoEndpoint extends Endpoint {
	private static Set<Session> sessions = Collections
			.synchronizedSet(new HashSet<Session>());

	public static synchronized void broadcast(String message) {
		for (Session session : sessions) {
			session.getAsyncRemote().sendText(message);
		}
	}

	@Override
	public void onClose(Session session, CloseReason closeReason) {
		sessions.remove(session);
	}

	@Override
	public void onOpen(final Session session, EndpointConfig config) {
		for (Monitor monitor : Monitor.getMonitors()) {
			session.getAsyncRemote().sendText(monitor.getValueString());
		}
		sessions.add(session);
	}
}
