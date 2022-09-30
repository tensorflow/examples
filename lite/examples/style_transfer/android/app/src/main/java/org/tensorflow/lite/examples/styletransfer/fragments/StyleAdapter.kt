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
package org.tensorflow.lite.examples.styletransfer.fragments

import android.annotation.SuppressLint
import android.net.Uri
import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.bumptech.glide.Glide
import org.tensorflow.lite.examples.styletransfer.databinding.ItemStyleBinding

class StyleAdapter(
    private val styles: MutableList<Style>, private val
    selected: (Int) -> Unit
) : RecyclerView
.Adapter<StyleAdapter.StyleViewHolder>() {
    private var previousPosition = RecyclerView.NO_POSITION

    @SuppressLint("NotifyDataSetChanged")
    fun setSelected(position: Int, isSelected: Boolean) {
        styles[position].isSelected = isSelected
        previousPosition = position
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(
        parent: ViewGroup,
        viewType: Int
    ): StyleViewHolder {
        val binding = ItemStyleBinding.inflate(
            LayoutInflater.from(parent.context),
            parent,
            false
        )
        return StyleViewHolder(binding)
    }

    override fun onBindViewHolder(holder: StyleViewHolder, position: Int) {
        holder.bind(styles[position])
    }

    override fun getItemCount(): Int = styles.size

    @SuppressLint("NotifyDataSetChanged")
    inner class StyleViewHolder(private val binding: ItemStyleBinding) :
        RecyclerView.ViewHolder(binding.root) {
        init {
            binding.imgStyle.setOnClickListener {
                selected.invoke(adapterPosition)
                styles[adapterPosition].isSelected = true
                if (previousPosition > RecyclerView.NO_POSITION) {
                    styles[previousPosition].isSelected = false
                }
                previousPosition = adapterPosition
                notifyDataSetChanged()
            }
        }

        fun bind(style: Style) {
            binding.root.isSelected = style.isSelected
            Glide.with(binding.root.context)
                .load(Uri.parse("file:///android_asset/thumbnails/${style.imagePath}"))
                .centerCrop().into(binding.imgStyle)
        }
    }

    data class Style(val imagePath: String, var isSelected: Boolean = false)
}
