val libs = libraries
val versionCatalog = extensions.getByType<VersionCatalogsExtension>().named("libs")

plugins {
    //trick: for the same plugin versions in all sub-modules
    id("com.android.application").version("7.4.2").apply(false)
    id("com.android.library").version("7.4.2").apply(false)
    kotlin("android").version("1.8.10").apply(false)
    id("com.android.test").version("7.4.0").apply(false)
    id("de.undercouch.download").version("4.0.2").apply(false)
}

tasks.register("clean", Delete::class) {
    delete(rootProject.buildDir)
}
