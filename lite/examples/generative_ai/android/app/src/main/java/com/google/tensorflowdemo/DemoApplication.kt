package com.google.tensorflowdemo

import android.app.Application
import com.google.tensorflowdemo.di.appModule
import com.google.tensorflowdemo.di.viewmodelModule
import io.github.aakira.napier.DebugAntilog
import io.github.aakira.napier.Napier
import org.koin.android.ext.koin.androidContext
import org.koin.android.ext.koin.androidLogger
import org.koin.core.context.startKoin

class DemoApplication : Application() {

    override fun onCreate() {
        super.onCreate()

        if (BuildConfig.DEBUG) {
            Napier.base(DebugAntilog())
        }

        startKoin {
            androidLogger()
            androidContext(this@DemoApplication)
            modules(
                appModule,
                viewmodelModule
            )
        }
    }
}