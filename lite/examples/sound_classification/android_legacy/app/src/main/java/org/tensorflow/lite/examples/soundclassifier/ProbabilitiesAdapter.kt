/*
 * Copyright 2020 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.soundclassifier

import android.animation.ObjectAnimator
import android.content.res.ColorStateList
import android.view.LayoutInflater
import android.view.ViewGroup
import android.view.animation.AccelerateDecelerateInterpolator
import androidx.recyclerview.widget.RecyclerView
import org.tensorflow.lite.examples.soundclassifier.databinding.ItemProbabilityBinding

internal class ProbabilitiesAdapter : RecyclerView.Adapter<ProbabilitiesAdapter.ViewHolder>() {
  var labelList = emptyList<String>()
  var probabilityMap = mapOf<String, Float>()

  override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
    val binding =
      ItemProbabilityBinding.inflate(LayoutInflater.from(parent.context), parent, false)
    return ViewHolder(binding)
  }

  override fun onBindViewHolder(holder: ViewHolder, position: Int) {
    val label = labelList[position]
    val probability = probabilityMap[label] ?: 0f
    holder.bind(position, label, probability)
  }

  override fun getItemCount() = labelList.size

  class ViewHolder(private val binding: ItemProbabilityBinding) :
    RecyclerView.ViewHolder(binding.root) {
    fun bind(position: Int, label: String, probability: Float) {
      with(binding) {
        labelTextView.text = label
        progressBar.progressBackgroundTintList = progressColorPairList[position % 3].first
        progressBar.progressTintList = progressColorPairList[position % 3].second

        val newValue = (probability * 100).toInt()
        // If you don't want to animate, you can write like `progressBar.progress = newValue`.
        val animation =
          ObjectAnimator.ofInt(progressBar, "progress", progressBar.progress, newValue)
        animation.duration = 100
        animation.interpolator = AccelerateDecelerateInterpolator()
        animation.start()
      }
    }

    companion object {
      /** List of pairs of background tint and progress tint */
      private val progressColorPairList = listOf(
        ColorStateList.valueOf(0xfff9e7e4.toInt()) to ColorStateList.valueOf(0xffd97c2e.toInt()),
        ColorStateList.valueOf(0xfff7e3e8.toInt()) to ColorStateList.valueOf(0xffc95670.toInt()),
        ColorStateList.valueOf(0xffecf0f9.toInt()) to ColorStateList.valueOf(0xff714Fe7.toInt()),
      )
    }
  }
}
