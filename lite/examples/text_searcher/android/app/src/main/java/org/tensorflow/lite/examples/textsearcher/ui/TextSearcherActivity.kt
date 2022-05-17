/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

package org.tensorflow.lite.examples.textsearcher.ui

import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageButton
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.widget.doOnTextChanged
import androidx.recyclerview.widget.DividerItemDecoration
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.textfield.TextInputLayout
import java.io.Serializable
import org.tensorflow.lite.examples.textsearcher.R
import org.tensorflow.lite.examples.textsearcher.tflite.Result
import org.tensorflow.lite.examples.textsearcher.tflite.TextSearcherClient

class MainActivity : AppCompatActivity() {
  private lateinit var presetQueryRecyclerView: RecyclerView
  private lateinit var searchButton: ImageButton
  private lateinit var searchQueryTextInput: TextInputLayout
  private lateinit var presetQueries: Array<String>
  private lateinit var presetQueryAdapter: PresetQueryAdapter
  private lateinit var textSearcherClient: TextSearcherClient

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_text_searcher)

    presetQueryRecyclerView = findViewById(R.id.preset_query_recycler_view)
    searchButton = findViewById(R.id.search_button)
    searchQueryTextInput = findViewById(R.id.search_query_text_input)
    textSearcherClient = TextSearcherClient.create(this)

    // Load the preset queries into an array.
    presetQueries = resources.getStringArray(R.array.tfe_preset_queries)

    presetQueryAdapter = PresetQueryAdapter(presetQueries) { startSearch(presetQueries[it]) }
    searchButton.setOnClickListener {
      val query = searchQueryTextInput.editText?.text.toString()
      startSearch(query)
    }
    initPresetQueryRecyclerView()
    updateSearchButton()
  }

  // Start the text search logic.
  private fun startSearch(queryContent: String) {
    val results = textSearcherClient.search(queryContent)
    openTextSearcherScreen(queryContent, results)
  }

  // Send search result to result activity.
  private fun openTextSearcherScreen(queryContent: String, results: List<Result>) {
    startActivity(
      Intent(this, TextSearcherResultActivity::class.java).apply {
        putExtra(getString(R.string.tfe_search_query_content), queryContent)
        putExtra(getString(R.string.tfe_search_result_id), results as Serializable)
      }
    )
  }

  // Set up the RecyclerView that shows the preset search queries.
  private fun initPresetQueryRecyclerView() {
    presetQueryRecyclerView.layoutManager = LinearLayoutManager(this)
    presetQueryRecyclerView.addItemDecoration(
      DividerItemDecoration(this, DividerItemDecoration.VERTICAL)
    )
    presetQueryRecyclerView.adapter = presetQueryAdapter
  }

  // Update the search button status.
  private fun updateSearchButton() {
    searchButton.isEnabled = false
    searchQueryTextInput.editText?.doOnTextChanged { text, _, _, _ ->
      // Only allow search if the search query isn't empty.
      searchButton.isEnabled = text?.trim()?.length ?: 0 > 0
    }
  }

  override fun onDestroy() {
    super.onDestroy()
    textSearcherClient.close()
  }
}

class PresetQueryAdapter(private val preset: Array<String>, private val onClick: (Int) -> Unit) :
  RecyclerView.Adapter<PresetQueryAdapter.PresetQueryViewHolder>() {

  inner class PresetQueryViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
    private val tvPreset: TextView = itemView.findViewById(R.id.tvPreset)

    init {
      itemView.setOnClickListener {
        if (adapterPosition == RecyclerView.NO_POSITION) return@setOnClickListener
        onClick.invoke(adapterPosition)
      }
    }

    fun bind(content: String) {
      tvPreset.text = content
    }
  }

  override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): PresetQueryViewHolder {
    val view =
      LayoutInflater.from(parent.context).inflate(R.layout.preset_query_item_layout, parent, false)

    return PresetQueryViewHolder(view)
  }

  override fun onBindViewHolder(holder: PresetQueryViewHolder, position: Int) {
    holder.bind(preset[position])
  }

  override fun getItemCount(): Int {
    return preset.size
  }
}
