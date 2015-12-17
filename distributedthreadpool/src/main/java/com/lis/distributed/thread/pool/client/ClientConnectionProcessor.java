package com.lis.distributed.thread.pool.client;

import com.lis.distributed.thread.pool.SocketAccesor;
import com.lis.distributed.thread.pool.server.ThreadPoolServer;

import java.io.IOException;
import java.net.Socket;

public class ClientConnectionProcessor extends SocketAccesor<ThreadPoolClient, ThreadPoolServer> {

    public ClientConnectionProcessor(ThreadPoolClient context, Socket accept)
            throws IOException {
        super(context, accept);
    }
}
