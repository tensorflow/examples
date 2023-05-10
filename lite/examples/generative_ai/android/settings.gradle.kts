@file:Suppress("UnstableApiUsage")
pluginManagement {
    repositories {
        google()
        gradlePluginPortal()
        mavenCentral()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.PREFER_SETTINGS)
    repositories {
        google()
        mavenCentral()
        maven {
            url = uri("https://jitpack.io")
        }
    }
    versionCatalogs {
        create("libraries") {
            from(files("gradle/libs.versions.toml"))
        }
    }
}

rootProject.name = "Google TensorFlow Demo"
include(":app")