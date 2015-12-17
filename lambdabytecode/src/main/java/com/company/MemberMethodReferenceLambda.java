package com.company;

public class MemberMethodReferenceLambda {

    public static void main(String[] args) {
        LambdaConstant.INTEGERS.stream().forEach(System.out::println);
    }

}
