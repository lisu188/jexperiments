package hrab;


import java.io.Serializable;

public class DayForecast implements Serializable {
    private double pressure;
    private int humidity;
    private double windSpeed;
    private double windDirection;

    private double high;
    private double low;

    private String description;
    private int weatherId;

    public DayForecast(double pressure, int humidity, double windSpeed,
                       double windDirection, double high, double low, String description,
                       int weatherId) {
        this.pressure = pressure;
        this.humidity = humidity;
        this.windSpeed = windSpeed;
        this.windDirection = windDirection;
        this.high = high;
        this.low = low;
        this.description = description;
        this.weatherId = weatherId;
    }

    public double getPressure() {
        return pressure;
    }

    public int getHumidity() {
        return humidity;
    }

    public double getWindSpeed() {
        return windSpeed;
    }

    public double getWindDirection() {
        return windDirection;
    }

    public double getHigh() {
        return high;
    }

    public double getLow() {
        return low;
    }

    public String getDescription() {
        return description;
    }

    public int getWeatherId() {
        return weatherId;
    }

    public String toString() {
        return (pressure + " " + humidity + " " + windSpeed + " "
                + windDirection + " " + description + " " + weatherId + " "
                + high + " " + low);
    }
}
