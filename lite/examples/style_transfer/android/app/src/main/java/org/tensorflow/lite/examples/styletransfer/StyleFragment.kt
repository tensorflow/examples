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

package org.tensorflow.lite.examples.styletransfer

import android.content.Context
import android.os.Bundle
import androidx.fragment.app.DialogFragment
import androidx.recyclerview.widget.GridLayoutManager
import androidx.recyclerview.widget.RecyclerView
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup

/**
 * A fragment representing a list of available Styles to apply
 * Activities containing this fragment MUST implement the
 * [StyleFragment.OnListFragmentInteractionListener] interface.
 */
class StyleFragment : DialogFragment() {

  private var listener: OnListFragmentInteractionListener? = null

  override fun onCreateView(
    inflater: LayoutInflater,
    container: ViewGroup?,
    savedInstanceState: Bundle?
  ): View? {
    val view = inflater.inflate(R.layout.fragment_style_list, container, false)

    val styles = ArrayList<String>()
    styles.addAll(activity!!.assets.list("thumbnails")!!)

    // Set the adapter
    if (view is RecyclerView) {
      with(view) {
        layoutManager = GridLayoutManager(context, 3)
        adapter = StyleRecyclerViewAdapter(styles, context, listener)
      }
    }
    return view
  }

  override fun onAttach(context: Context) {
    super.onAttach(context)
    if (context is OnListFragmentInteractionListener) {
      listener = context
    } else {
      throw RuntimeException("$context must implement OnListFragmentInteractionListener")
    }
  }

  override fun onDetach() {
    super.onDetach()
    listener = null
  }

  /**
   * This interface must be implemented by activities that contain this
   * fragment to allow an interaction in this fragment to be communicated
   * to the activity and potentially other fragments contained in that
   * activity.
   *
   *
   * See the Android Training lesson
   * [Communicating with Other Fragments](http://developer.android.com/training/basics/fragments/communicating.html)
   * for more information.
   */
  interface OnListFragmentInteractionListener {
    fun onListFragmentInteraction(item: String)
  }
}
