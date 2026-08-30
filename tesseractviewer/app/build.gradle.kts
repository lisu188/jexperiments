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
        versionCode = 1
        versionName = "1.0"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildTypes {
        release {
            isMinifyEnabled = true
        }
    }
}

dependencies {
    testImplementation("junit:junit:4.13.2")
}
