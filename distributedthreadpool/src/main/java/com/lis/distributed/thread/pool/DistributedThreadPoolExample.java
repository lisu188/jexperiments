package com.lis.distributed.thread.pool;

import com.lis.distributed.thread.pool.client.ThreadPoolClient;
import com.lis.distributed.thread.pool.server.ThreadPoolServer;

import java.time.Duration;
import java.util.ArrayList;

public final class DistributedThreadPoolExample {
    public static void main(String[] args) throws Exception {
        try (var server = new ThreadPoolServer(0).start();
             var client = new ThreadPoolClient("127.0.0.1", server.port())) {
            var clientId = client.awaitClientId(Duration.ofSeconds(5));
            System.out.println("client id = " + clientId);

            var futures = new ArrayList<java.util.concurrent.CompletableFuture<Integer>>();
            for (int i = 0; i < 8; i++) {
                var value = i;
                futures.add(client.callOnServer(() -> value * value));
            }
            System.out.println("server results = " + futures.stream().map(java.util.concurrent.CompletableFuture::join).toList());

            System.out.println("client result = "
                    + server.callOnClient(clientId, () -> "hello from client", Duration.ofSeconds(5)));

            client.executeOnServer(context -> {
                if (context.clientCount() != 1) {
                    throw new IllegalStateException("unexpected client count");
                }
            }).join();

            System.out.println("client transport = " + client.statistics());
        }
    }
}
