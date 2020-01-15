/*
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.imagesegmentation

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import android.content.Context
import java.io.File
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch

class MLExecutionViewModel : ViewModel() {

  lateinit var imageSegmentationModel: ImageSegmentationModelExecutor
  private val _styledBitmap = MutableLiveData<ModelExecutionResult>()

  val styledBitmap: LiveData<ModelExecutionResult>
    get() = _styledBitmap

  private val viewModelJob = Job()
  private val viewModelScope = CoroutineScope(viewModelJob)

  fun onApplyStyle(
    context: Context,
    filePath: String,
    useGpu: Boolean = false
  ) {
    viewModelScope.launch(Dispatchers.Default) {
      imageSegmentationModel =
        ImageSegmentationModelExecutor(
          context,
          useGpu
        )
      val contentImage =
        ImageUtils.decodeBitmap(
          File(filePath)
        )

      val result = imageSegmentationModel.execute(contentImage)
      _styledBitmap.postValue(result)
      imageSegmentationModel.close()
    }
  }

  override fun onCleared() {
    super.onCleared()
    imageSegmentationModel.close()
  }
}
