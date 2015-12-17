package hrab;

import java.io.Serializable;

/**
 * Created by zlatan on 2015-03-25.
 */
public class WeatherQuery implements Serializable {
    private String location;
    private String units;

    public WeatherQuery(String location, String units) {
        this.location = location;
        this.units = units;
    }

    public String getLocation() {
        return location;
    }

    public String getUnits() {
        return units;
    }
}
