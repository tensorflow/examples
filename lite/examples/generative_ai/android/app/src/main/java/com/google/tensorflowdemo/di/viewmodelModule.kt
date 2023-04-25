package com.google.tensorflowdemo.di

import com.google.tensorflowdemo.ui.screens.autocomplete.AutoCompleteViewModel
import org.koin.androidx.viewmodel.dsl.viewModelOf
import org.koin.dsl.module

val viewmodelModule = module {
    viewModelOf(::AutoCompleteViewModel)
}