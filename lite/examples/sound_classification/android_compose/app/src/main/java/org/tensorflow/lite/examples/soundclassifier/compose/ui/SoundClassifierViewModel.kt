package org.tensorflow.lite.examples.soundclassifier.compose.ui

import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.tensorflow.lite.support.label.Category

const val DefaultClassificationInterval = 500L

class SoundClassifierViewModel : ViewModel() {
  // How often should classification run in milliseconds
  private val _classifierEnabled = MutableStateFlow(true)
  val classifierEnabled = _classifierEnabled.asStateFlow()

  // How often should classification run in milliseconds
  private val _classificationInterval = MutableStateFlow(DefaultClassificationInterval)
  val classificationInterval = _classificationInterval.asStateFlow()

  // As a result of sound classification, this value emits map of probabilities
  private val _probabilities = MutableStateFlow<List<Category>>(emptyList())
  val probabilities = _probabilities.asStateFlow()

  fun setClassifierEnabled(value: Boolean) {
    _classifierEnabled.value = value
  }

  fun setClassificationInterval(value: Long) {
    _classificationInterval.value = value
  }

  fun setProbabilities(value: List<Category>) {
    _probabilities.value = value
  }
}