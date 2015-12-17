package hrab;


import java.io.Serializable;
import java.util.List;

public class WeatherMessage implements Serializable {
	private String cityName;
	private double longitude;
	private double latitude;

	public WeatherMessage(String cityName, double longitude, double latitude,
			List<DayForecast> daysForecast) {
		this.cityName = cityName;
		this.longitude = longitude;
		this.latitude = latitude;
		this.daysForecast = daysForecast;
	}

	public String getCityName() {
		return cityName;
	}

	public double getLongitude() {
		return longitude;
	}

	public double getLatitude() {
		return latitude;
	}

	public List<DayForecast> getDaysForecast() {
		return daysForecast;
	}

	private List<DayForecast> daysForecast;

	public String toString() {
		String s = cityName + "\n";
		s += latitude + "\n";
		s += longitude + "\n";
		s += daysForecast;
		return s;
	}
}
