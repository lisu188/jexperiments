package com.company;

public class ExpressionLambda {

    public static void main(String[] args) {
        LambdaConstant.INTEGERS.stream().forEach(x -> {
            System.out.println(x);
        });
    }

}
