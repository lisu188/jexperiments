package com.company;

import java.util.Arrays;
import java.util.List;

public class LambdaConstant {

    public static final List<Integer> INTEGERS = Arrays.asList(1, 2, 3, 4, 5);

    public static void println(Integer x) {
        System.out.println(x);
    }

    public static class Constructor {
        public Constructor(Integer x) {
            System.out.println(x);
        }
    }
}
