package com.lis.pi.websocket;

import javax.websocket.Decoder;
import javax.websocket.Encoder;
import javax.websocket.Extension;
import javax.websocket.server.ServerEndpointConfig;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class InfoEndpointConfig implements ServerEndpointConfig {

    private final Configurator configurator = new Configurator();
    private final Map<String, Object> userProperties = new ConcurrentHashMap<String, Object>();

    @Override
    public Configurator getConfigurator() {
        return configurator;
    }

    @Override
    public List<Class<? extends Decoder>> getDecoders() {
        return new ArrayList<Class<? extends Decoder>>();
    }

    @Override
    public List<Class<? extends Encoder>> getEncoders() {
        return new ArrayList<Class<? extends Encoder>>();
    }

    @Override
    public Class<?> getEndpointClass() {
        return InfoEndpoint.class;
    }

    @Override
    public List<Extension> getExtensions() {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public String getPath() {
        return "/info";
    }

    @Override
    public List<String> getSubprotocols() {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public Map<String, Object> getUserProperties() {
        return userProperties;
    }

}
