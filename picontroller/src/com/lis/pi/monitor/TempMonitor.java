package com.lis.pi.monitor;

import java.io.IOException;

import org.apache.logging.log4j.LogManager;

import com.pi4j.system.SystemInfo;

public class TempMonitor extends Monitor {
	public TempMonitor() {
		super(ValueType.TEMPERATURE);
	}

	@Override
	public void run() {
		while (!terminated) {
			try {
				setValue(SystemInfo.getCpuTemperature());
				Thread.sleep(1000);
			} catch (NumberFormatException | IOException | InterruptedException e) {
				LogManager.getLogger(getClass()).error("Error", e);
			}
		}
	}

}
