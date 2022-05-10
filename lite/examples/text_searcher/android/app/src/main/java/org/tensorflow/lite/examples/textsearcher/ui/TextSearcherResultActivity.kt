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
import android.view.*
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.DividerItemDecoration
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import org.tensorflow.lite.examples.textsearcher.R
import org.tensorflow.lite.examples.textsearcher.tflite.Result

class TextSearcherResultActivity : AppCompatActivity() {
  private lateinit var searchQueryTextView: TextView
  private lateinit var searchResultRecyclerView: RecyclerView
  private lateinit var searchResultAdapter: SearchResultAdapter

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_text_searcher_result)
    searchQueryTextView = findViewById(R.id.tvQueryContent)
    searchResultRecyclerView = findViewById(R.id.search_result_recycler_view)

    val results =
      intent.getSerializableExtra(getString(R.string.tfe_search_result_id)) as List<Result>
    val queryContent = intent.getStringExtra(getString(R.string.tfe_search_query_content))

    searchQueryTextView.text = queryContent
    searchResultAdapter =
      SearchResultAdapter(results) { position -> openSearchResultUrl(results[position].url) }
    initSearchResultRecyclerView()
  }

  // Open a search result's metadata URL in an webview.
  private fun openSearchResultUrl(url: String) {
    startActivity(
      Intent(this, WebviewActivity::class.java).apply {
        putExtra(getString(R.string.tfe_target_url), url)
      }
    )
  }

  // Initialize the search result RecyclerView.
  private fun initSearchResultRecyclerView() {
    searchResultRecyclerView.layoutManager = LinearLayoutManager(this)
    searchResultRecyclerView.addItemDecoration(
      DividerItemDecoration(this, DividerItemDecoration.VERTICAL)
    )
    searchResultRecyclerView.adapter = searchResultAdapter
  }

  override fun onCreateOptionsMenu(menu: Menu?): Boolean {
    menuInflater.inflate(R.menu.action_bar_menu, menu)
    return true
  }

  override fun onOptionsItemSelected(item: MenuItem): Boolean =
    when (item.itemId) {
      R.id.action_close -> {
        // The user chose the "Close" action, so close this activity.
        finish()
        true
      }
      else -> {
        // If we got here, the user's action was not recognized.
        // Invoke the superclass to handle it.
        super.onOptionsItemSelected(item)
      }
    }
}

class SearchResultAdapter(private val results: List<Result>, private val onClick: (Int) -> Unit) :
  RecyclerView.Adapter<SearchResultAdapter.SearchResultViewHolder>() {

  inner class SearchResultViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
    private val distanceTextView: TextView = itemView.findViewById(R.id.distance_text_view)
    private val urlTextView: TextView = itemView.findViewById(R.id.url_text_view)

    init {
      itemView.setOnClickListener {
        if (adapterPosition == RecyclerView.NO_POSITION) return@setOnClickListener
        onClick.invoke(adapterPosition)
      }
    }

    fun bind(result: Result) {
      distanceTextView.text =
        itemView.context.getString(R.string.tfe_tv_distance, result.distance.toString())
      urlTextView.text = result.url
    }
  }

  override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): SearchResultViewHolder {
    val view =
      LayoutInflater.from(parent.context)
        .inflate(R.layout.searcher_result_item_layout, parent, false)
    return SearchResultViewHolder(view)
  }

  override fun onBindViewHolder(holder: SearchResultViewHolder, position: Int) {
    holder.bind(results[position])
  }

  override fun getItemCount(): Int {
    return results.size
  }
}
