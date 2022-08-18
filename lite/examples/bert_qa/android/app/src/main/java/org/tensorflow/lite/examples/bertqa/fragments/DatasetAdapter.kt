/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.tensorflow.lite.examples.bertqa.fragments

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import org.tensorflow.lite.examples.bertqa.databinding.ItemDatasetBinding

class DatasetAdapter(
    private val dataSetTitle: List<String>,
    private val onSelected: (Int) -> Unit
) :
    RecyclerView.Adapter<DatasetAdapter.ViewHolder>() {

    inner class ViewHolder(private val binding: ItemDatasetBinding) :
        RecyclerView.ViewHolder(binding.root) {
        init {
            binding.tvDataTitle.setOnClickListener {
                onSelected.invoke(adapterPosition)
            }
        }

        fun bind(title: String) {
            with(binding) {
                tvDataTitle.text = title
            }
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val binding = ItemDatasetBinding.inflate(
            LayoutInflater.from(parent.context),
            parent,
            false
        )
        return ViewHolder(binding)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        holder.bind(dataSetTitle[position])
    }

    override fun getItemCount() = dataSetTitle.size
}
