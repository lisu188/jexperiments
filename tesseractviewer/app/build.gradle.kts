plugins {
    id("com.android.application")
}

android {
    namespace = "experiments.tesseractviewer"
    compileSdk = 36

    defaultConfig {
        applicationId = "experiments.tesseractviewer"
        minSdk = 26
        targetSdk = 36
        versionCode = 3
        versionName = "1.1.0"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildTypes {
        release {
            isDebuggable = false
            isMinifyEnabled = true
            isShrinkResources = true
            signingConfig = signingConfigs.getByName("debug")
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"))
        }
    }
}

dependencies {
    testImplementation("junit:junit:4.13.2")
}
