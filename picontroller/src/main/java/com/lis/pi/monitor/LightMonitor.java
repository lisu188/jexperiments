package com.lis.pi.monitor;

import com.pi4j.io.gpio.GpioFactory;
import com.pi4j.io.gpio.GpioPinDigitalInput;
import com.pi4j.io.gpio.PinPullResistance;
import com.pi4j.io.gpio.RaspiPin;
import com.pi4j.io.gpio.event.GpioPinDigitalStateChangeEvent;
import com.pi4j.io.gpio.event.GpioPinListenerDigital;

public class LightMonitor extends Monitor {
    private final GpioPinDigitalInput lightPin1;
    private final GpioPinDigitalInput lightPin2;

    public LightMonitor() {
        super(ValueType.LIGHT);
        lightPin1 = GpioFactory.getInstance().provisionDigitalInputPin(
                RaspiPin.GPIO_08, PinPullResistance.PULL_DOWN);
        lightPin2 = GpioFactory.getInstance().provisionDigitalInputPin(
                RaspiPin.GPIO_09, PinPullResistance.PULL_DOWN);
        GpioPinListenerDigital listener = new GpioPinListenerDigital() {

            @Override
            public void handleGpioPinDigitalStateChangeEvent(
                    GpioPinDigitalStateChangeEvent paramGpioPinDigitalStateChangeEvent) {
                setValue(getLightState());
            }

        };
        lightPin1.addListener(listener);
        lightPin2.addListener(listener);
        setValue(getLightState());
    }

    private int getLightState() {
        if (lightPin1.getState().isHigh() && lightPin2.getState().isHigh()) {
            return 2;
        } else if (!lightPin1.getState().isHigh()
                && !lightPin2.getState().isHigh()) {
            return 0;
        } else {
            return 1;
        }
    }
}
