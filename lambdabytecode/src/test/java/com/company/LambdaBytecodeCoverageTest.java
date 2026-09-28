package com.company;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertNotNull;

class LambdaBytecodeCoverageTest {
    @Test
    void constructorsAndLambdaHelpersAreCovered() {
        assertNotNull(new ConstructorReferenceLambda());
        assertNotNull(new ExpressionLambda());
        assertNotNull(new MemberMethodReferenceLambda());
        assertNotNull(new StaticMethodReferenceLambda());
        assertNotNull(new LambdaConstant());
        assertNotNull(new LambdaConstant.Constructor(99));

        LambdaConstant.println(100);
        ConstructorReferenceLambda.main(new String[0]);
        ExpressionLambda.main(new String[0]);
        MemberMethodReferenceLambda.main(new String[0]);
        StaticMethodReferenceLambda.main(new String[0]);
    }
}
