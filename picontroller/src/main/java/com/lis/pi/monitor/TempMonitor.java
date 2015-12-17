package com.lis.pi.monitor;

import com.pi4j.system.SystemInfo;
import org.apache.logging.log4j.LogManager;

import java.io.IOException;

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
