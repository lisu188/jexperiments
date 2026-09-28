package com.lis.pi.monitor;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MonitorDataTest {
    private MonitorData data;

    @BeforeEach
    void reset() {
        data = MonitorData.getInstance();
        data.getPolledData().clear();
    }

    @Test
    void singletonStoresAndMutatesDataUnits() {
        assertSame(data, MonitorData.getInstance());

        data.updateData(ValueType.LIGHT, 1.5f);
        data.updateData(ValueType.MEMORY, 2.5f);
        data.updateData(ValueType.TEMPERATURE, 3.5f);

        assertEquals(3, data.getPolledData().size());
        MonitorData.DataUnit unit = data.getPolledData().get(0);
        assertEquals(ValueType.LIGHT, unit.getType());
        assertEquals(1.5f, unit.getValue());
        assertTrue(unit.getTimeStamp() > 0);

        unit.setType(ValueType.MEMORY);
        unit.setValue(7.25f);
        unit.setTimeStamp(123L);

        assertEquals(ValueType.MEMORY, unit.getType());
        assertEquals(7.25f, unit.getValue());
        assertEquals(123L, unit.getTimeStamp());
    }

    @Test
    void historyIsPurgedAfterTenThousandSamples() {
        for (int i = 0; i <= 10_000; i++) {
            data.updateData(ValueType.LIGHT, i);
        }
        assertTrue(data.getPolledData().isEmpty());
    }
}
