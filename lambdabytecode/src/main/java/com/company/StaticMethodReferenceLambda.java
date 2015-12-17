package com.company;

public class StaticMethodReferenceLambda {

    public static void main(String[] args) {
        LambdaConstant.INTEGERS.stream().forEach(LambdaConstant::println);
    }

}
