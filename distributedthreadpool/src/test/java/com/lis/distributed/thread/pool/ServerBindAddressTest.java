package com.lis.distributed.thread.pool;

import com.lis.distributed.thread.pool.client.ThreadPoolClient;
import com.lis.distributed.thread.pool.server.ThreadPoolServer;
import org.junit.jupiter.api.Test;

import java.net.InetAddress;
import java.time.Duration;
import java.util.concurrent.Executors;

import static org.junit.jupiter.api.Assertions.*;

class ServerBindAddressTest {
    @Test
    void defaultListenerIsLoopbackAndStillExecutesRequests() throws Exception {
        try (var server = new ThreadPoolServer(0)) {
            assertThrows(IllegalStateException.class, server::address);
            server.start();
            assertTrue(server.address().isLoopbackAddress());
            assertFalse(server.address().isAnyLocalAddress());
            try (var client = new ThreadPoolClient(server.address().getHostAddress(), server.port())) {
                client.awaitClientId(Duration.ofSeconds(5));
                assertEquals(42, client.callOnServer(() -> 42, Duration.ofSeconds(5)));
            }
        }
    }

    @Test
    void suppliedExecutorAlsoDefaultsToLoopback() throws Exception {
        try (var executor = Executors.newSingleThreadExecutor();
             var server = new ThreadPoolServer(0, executor, SocketAccessor.Options.defaults()).start()) {
            assertTrue(server.address().isLoopbackAddress());
        }
    }

    @Test
    void wildcardExposureRequiresAnExplicitBindAddress() throws Exception {
        var wildcard = InetAddress.getByAddress(new byte[4]);
        try (var server = new ThreadPoolServer(0, wildcard).start()) {
            assertTrue(server.address().isAnyLocalAddress());
        }
        try (var executor = Executors.newSingleThreadExecutor();
             var server = new ThreadPoolServer(0, InetAddress.getLoopbackAddress(), executor,
                     SocketAccessor.Options.defaults()).start()) {
            assertTrue(server.address().isLoopbackAddress());
        }
    }

    @Test
    void nullBindAddressCannotSilentlySelectWildcard() {
        assertThrows(NullPointerException.class, () -> new ThreadPoolServer(0, (InetAddress) null));
        try (var executor = Executors.newSingleThreadExecutor()) {
            assertThrows(NullPointerException.class,
                    () -> new ThreadPoolServer(0, null, executor, SocketAccessor.Options.defaults()));
        }
    }
}
