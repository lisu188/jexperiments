package com.lis.pi.monitor;

import java.io.IOException;

import org.apache.logging.log4j.LogManager;

import com.pi4j.system.SystemInfo;

public class MemMonitor extends Monitor {
	public MemMonitor() {
		super(ValueType.MEMORY);
	}

	@Override
	public void run() {
		while (!terminated) {
			try {
				float value = (float) SystemInfo.getMemoryUsed()
						/ (float) SystemInfo.getMemoryTotal();
				setValue((float) ((double) Math.round(value * 100) / 100));
				Thread.sleep(1000);
			} catch (NumberFormatException | IOException | InterruptedException e) {
				LogManager.getLogger(getClass()).error("Error", e);
			}
		}
	}
}
