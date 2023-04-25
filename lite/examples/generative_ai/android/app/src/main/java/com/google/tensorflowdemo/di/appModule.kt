package com.google.tensorflowdemo.di

import com.google.tensorflowdemo.data.autocomplete.AutoCompleteService
import com.google.tensorflowdemo.data.autocomplete.AutoCompleteServiceImpl
import com.mediamonks.wordfilter.LanguageChecker
import com.mediamonks.wordfilter.LanguageCheckerImpl
import org.koin.android.ext.koin.androidContext
import org.koin.core.module.dsl.singleOf
import org.koin.dsl.module

val appModule = module {
    single<AutoCompleteService> {
        AutoCompleteServiceImpl(
            context = androidContext(),
            languageChecker = get()
        )
    }

    singleOf<LanguageChecker>(::LanguageCheckerImpl)
}
