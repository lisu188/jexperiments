package com.lis.pi;

import org.apache.logging.log4j.LogManager;

import com.lis.pi.monitor.Monitor;

public class MainClass {
	public static void main(String[] args) {
		try {
			Monitor.startMonitors();
			TomcatManager.getInstance().getTomcat().start();
			TomcatManager.getInstance().getTomcat().getServer().await();
			Monitor.stopMonitors();
		} catch (Exception e) {
			LogManager.getLogger(MainClass.class).error("Error", e);
		}
	}
}