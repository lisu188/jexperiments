plugins {
    id("com.android.application")
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(27)
    }
}

android {
    namespace = "experiments.tesseractviewer"
    compileSdk = 36

    defaultConfig {
        applicationId = "experiments.tesseractviewer"
        minSdk = 26
        targetSdk = 36
        versionCode = 4
        versionName = "1.1.1"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildTypes {
        debug {
            enableUnitTestCoverage = true
        }
        release {
            isDebuggable = false
            isMinifyEnabled = true
            isShrinkResources = true
            signingConfig = signingConfigs.getByName("debug")
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"))
        }
    }

    testCoverage {
        jacocoVersion = "0.8.15"
    }
}

dependencies {
    testImplementation("junit:junit:4.13.2")
}
