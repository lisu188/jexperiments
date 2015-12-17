package com.lis.distributed.thread.pool.server;

import com.lis.distributed.thread.pool.DataRepository;
import com.lis.distributed.thread.pool.client.ThreadPoolClient;
import com.lis.distributed.thread.pool.func.SerializableConsumer;
import com.lis.distributed.thread.pool.func.SerializableSupplier;

import java.net.ServerSocket;
import java.util.concurrent.*;

public class ThreadPoolServer {
    private ExecutorService connections = Executors.newCachedThreadPool();

    private int port;

    private ConcurrentHashMap<Integer, ServerConnectionThread> clients = new ConcurrentHashMap<Integer, ServerConnectionThread>();

    private DataRepository repository = new DataRepository();

    public ThreadPoolServer(int port) {
        this.port = port;
    }

    public static void main(String[] args) throws InterruptedException {
        new ThreadPoolServer(55555).start().await();
    }

    private ThreadPoolServer await() throws InterruptedException {
        connections.awaitTermination(1000, TimeUnit.SECONDS);
        return this;
    }

    private ThreadPoolServer start() {
        connections.submit(new Callable<Void>() {
            @Override
            public Void call() throws Exception {
                try (ServerSocket socket = new ServerSocket(port)) {
                    while (true) {
                        connections.submit(new ServerConnectionThread(
                                ThreadPoolServer.this, socket.accept()));
                    }
                }
            }
        });
        return this;
    }

    public void registerClient(int id,
                               ServerConnectionThread serverConnectionThread) {
        this.clients.put(id, serverConnectionThread);
    }

    public ServerConnectionThread getClient(int id) {
        return clients.get(id);
    }

    public <T, U> void callOnClient(int clientid,
                                    SerializableSupplier<T> target, SerializableConsumer<T> callback)
            throws Exception {
        callOnClient(clientid, (ctx) -> {
            T result = target.get();
            if (callback != null) {
                ctx.callOnServer((server) -> {
                    callback.accept(result);
                });
            }
        });
    }

    public <T, U> T callOnClient(int clientid, SerializableSupplier<T> target)
            throws Exception {
        int msgId = repository.lock();
        callOnClient(clientid, (ThreadPoolClient context) -> {
            T t = target.get();
            context.callOnServer((ThreadPoolServer ctx) -> {
                ctx.repository.setValue(msgId, t);
            });
        });
        return (T) repository.getValue(msgId);
    }

    public void callOnClient(int clientid,
                             SerializableConsumer<ThreadPoolClient> target) throws Exception {
        getClient(clientid).postMessage(target);
    }
}
