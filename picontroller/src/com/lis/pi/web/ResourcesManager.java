package com.lis.pi.web;

import java.io.InputStream;
import java.net.URL;

public class ResourcesManager {
	private static ResourcesManager instance = new ResourcesManager();

	public static ResourcesManager getInstance() {
		return instance;
	}

	private ResourcesManager() {
	}

	public URL getWebResource(String path) {
		return getClass().getResource(path);
	}

	public InputStream getWebResourceAsStream(String path) {
		return getClass().getResourceAsStream(path);
	}

}
