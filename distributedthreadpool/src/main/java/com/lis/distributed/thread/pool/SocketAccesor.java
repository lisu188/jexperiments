package com.lis.distributed.thread.pool;

import com.lis.distributed.thread.pool.func.SerializableConsumer;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.Socket;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class SocketAccesor<T, U> implements Callable<Void> {
    private T context;
    private ObjectOutputStream out;
    private ObjectInputStream in;
    private ExecutorService executor = Executors.newSingleThreadExecutor();

    public SocketAccesor(T context, Socket accept) throws IOException {
        this.context = context;
        out = new ObjectOutputStream(accept.getOutputStream());
        in = new ObjectInputStream(accept.getInputStream());
    }

    @Override
    public Void call() throws Exception {
        while (true) {
            Object readObject = in.readObject();
            if (readObject instanceof SerializableConsumer) {
                executor.submit((Callable<Void>) () -> {
                    ((SerializableConsumer<T>) readObject).accept(context);
                    return null;
                });
            }

        }
    }

    public void postMessage(SerializableConsumer<U> message) throws Exception {
        executor.submit((Callable<Void>) () -> {
            out.writeObject(message);
            return null;
        });
    }
}
