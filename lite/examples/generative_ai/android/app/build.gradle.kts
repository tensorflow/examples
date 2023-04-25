@file:Suppress("UnstableApiUsage")

plugins {
    kotlin("android")
    id("com.android.application")
    id("de.undercouch.download")
}

ext {
    set("AAR_URL", "https://storage.googleapis.com/download.tensorflow.org/models/tflite/generativeai/tensorflow-lite-select-tf-ops.aar")
    set("AAR_PATH", "$projectDir/libs/tensorflow-lite-select-tf-ops.aar")
}

apply {
    from("download.gradle")
}

android {
    namespace = "com.google.tensorflowdemo"
    compileSdk = 33

    defaultConfig {
        applicationId = "com.google.tensorflowdemo"
        minSdk = 24
        targetSdk = 33
        versionCode = 1
        versionName = "1.0"
    }
    buildFeatures {
        compose = true
        buildConfig = true
        viewBinding = true
    }
    composeOptions {
        kotlinCompilerExtensionVersion = "1.3.2"
    }
    packagingOptions {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
    buildTypes {
        getByName("release") {
            isMinifyEnabled = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            isDebuggable = false
        }
        getByName("debug") {
            applicationIdSuffix = ".debug"
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = "1.8"
        freeCompilerArgs = listOf(
            "-P",
            "plugin:androidx.compose.compiler.plugins.kotlin:suppressKotlinVersionCompatibilityCheck=1.8.10"
        )
    }
}

dependencies {
    implementation(fileTree(mapOf("dir" to "libs", "include" to listOf("*.aar"))))

    // Compose
    implementation(libraries.compose.ui)
    implementation(libraries.compose.ui.tooling)
    implementation(libraries.compose.ui.tooling.preview)
    implementation(libraries.compose.foundation)
    implementation(libraries.compose.material)
    implementation(libraries.compose.material.icons)
    implementation(libraries.compose.activity)

    // Accompanist for Compose
    implementation(libraries.accompanist.systemuicontroller)

    // Koin
    implementation(libraries.koin.core)
    implementation(libraries.koin.android)
    implementation(libraries.koin.compose)

    // Lifecycle
    implementation(libraries.lifecycle.viewmodel)
    implementation(libraries.lifecycle.viewmodel.compose)
    implementation(libraries.lifecycle.viewmodel.ktx)
    implementation(libraries.lifecycle.runtime.compose)

    // Logging
    implementation(libraries.napier)

    // Profanity filter
    implementation(libraries.wordfilter)

    // TensorFlow Lite
    implementation(libraries.tflite)

    // Unit tests
    testImplementation(libraries.junit)
}
