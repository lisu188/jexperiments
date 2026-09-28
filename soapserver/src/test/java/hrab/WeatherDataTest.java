package hrab;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class WeatherDataTest {
    @Test
    void forecastExposesAllValues() {
        DayForecast forecast = new DayForecast(1013.2, 61, 5.5, 220.0, 24.0, 12.0, "clear", 800);

        assertEquals(1013.2, forecast.getPressure());
        assertEquals(61, forecast.getHumidity());
        assertEquals(5.5, forecast.getWindSpeed());
        assertEquals(220.0, forecast.getWindDirection());
        assertEquals(24.0, forecast.getHigh());
        assertEquals(12.0, forecast.getLow());
        assertEquals("clear", forecast.getDescription());
        assertEquals(800, forecast.getWeatherId());
        assertTrue(forecast.toString().contains("clear"));
    }

    @Test
    void weatherMessageAndQueryExposePayload() {
        DayForecast forecast = new DayForecast(1000.0, 50, 3.0, 90.0, 20.0, 10.0, "cloudy", 801);
        WeatherMessage message = new WeatherMessage("Krakow", 19.94, 50.06, List.of(forecast));
        WeatherQuery query = new WeatherQuery("Krakow", "metric");

        assertEquals("Krakow", message.getCityName());
        assertEquals(19.94, message.getLongitude());
        assertEquals(50.06, message.getLatitude());
        assertEquals(List.of(forecast), message.getDaysForecast());
        assertTrue(message.toString().contains("Krakow"));
        assertEquals("Krakow", query.getLocation());
        assertEquals("metric", query.getUnits());

        WeatherEntry entry = new WeatherEntry();
        assertNotNull(entry);
        assertEquals("weather", WeatherEntry.TABLE_NAME);
        assertEquals("location_id", WeatherEntry.COLUMN_LOC_KEY);
        assertEquals("date", WeatherEntry.COLUMN_DATE);
        assertEquals("weather_id", WeatherEntry.COLUMN_WEATHER_ID);
        assertEquals("short_desc", WeatherEntry.COLUMN_SHORT_DESC);
        assertEquals("min", WeatherEntry.COLUMN_MIN_TEMP);
        assertEquals("max", WeatherEntry.COLUMN_MAX_TEMP);
        assertEquals("humidity", WeatherEntry.COLUMN_HUMIDITY);
        assertEquals("pressure", WeatherEntry.COLUMN_PRESSURE);
        assertEquals("wind", WeatherEntry.COLUMN_WIND_SPEED);
        assertEquals("degrees", WeatherEntry.COLUMN_DEGREES);
    }
}
