package com.company;

public class ConstructorReferenceLambda {

    public static void main(String[] args) {
        LambdaConstant.INTEGERS.stream().forEach(LambdaConstant.Constructor::new);
    }

}
