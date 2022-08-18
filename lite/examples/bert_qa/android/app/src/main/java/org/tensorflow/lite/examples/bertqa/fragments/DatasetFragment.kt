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

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.navigation.Navigation
import androidx.recyclerview.widget.DividerItemDecoration
import androidx.recyclerview.widget.LinearLayoutManager
import org.tensorflow.lite.examples.bertqa.R
import org.tensorflow.lite.examples.bertqa.databinding.FragmentDatasetBinding
import org.tensorflow.lite.examples.bertqa.dataset.LoadDataSetClient

class DatasetFragment : Fragment() {

    private var _fragmentDatasetList: FragmentDatasetBinding? = null
    private val fragmentDatasetList get() = _fragmentDatasetList!!
    private var titles : List<String> = emptyList()
    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _fragmentDatasetList = FragmentDatasetBinding.inflate(inflater, container, false)

        return fragmentDatasetList.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        val client = LoadDataSetClient(requireActivity())
        client.loadJson()?.let {
            titles = it.getTitles()
        }
        initRecyclerView()
    }

    private fun initRecyclerView() {
            val dataSetAdapter = DatasetAdapter(titles) {
                startQaScreen(it)
            }
            val linearLayoutManager = LinearLayoutManager(requireContext())
            val decoration = DividerItemDecoration(
                fragmentDatasetList.recyclerView.context,
                linearLayoutManager.orientation
            )
            with(fragmentDatasetList.recyclerView) {
                layoutManager = linearLayoutManager
                adapter = dataSetAdapter
                addItemDecoration(decoration)
            }
    }

    private fun startQaScreen(position: Int) {
        val action = DatasetFragmentDirections.actionDatasetFragmentToQaFragment(position)
        Navigation.findNavController(requireActivity(), R.id.fragment_container)
            .navigate(action)
    }
}
