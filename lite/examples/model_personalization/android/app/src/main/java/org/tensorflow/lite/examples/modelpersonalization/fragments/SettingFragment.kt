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

package org.tensorflow.lite.examples.modelpersonalization.fragments

import android.app.Dialog
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.DialogFragment
import androidx.fragment.app.activityViewModels
import org.tensorflow.lite.examples.modelpersonalization.MainViewModel
import org.tensorflow.lite.examples.modelpersonalization.databinding.FragmentSettingBinding


class SettingFragment : DialogFragment() {
    companion object {
        const val TAG = "SettingDialogFragment"
    }

    private var _fragmentSettingBinding: FragmentSettingBinding? = null
    private val fragmentSettingBinding
        get() = _fragmentSettingBinding!!

    private var numThreads = 2
    private val viewModel: MainViewModel by activityViewModels()

    override fun onCreateDialog(savedInstanceState: Bundle?): Dialog {
        return super.onCreateDialog(savedInstanceState).apply {
            setCancelable(false)
            setCanceledOnTouchOutside(false)
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {

        _fragmentSettingBinding = FragmentSettingBinding.inflate(
            inflater,
            container,
            false
        )
        return fragmentSettingBinding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        viewModel.getNumThreads()?.let {
            numThreads = it
            updateDialogUi()
        }

        initDialogControls()

        fragmentSettingBinding.btnConfirm.setOnClickListener {
            viewModel.configModel(numThreads)
            dismiss()
        }
        fragmentSettingBinding.btnCancel.setOnClickListener {
            dismiss()
        }
    }

    private fun initDialogControls() {
        // When clicked, decrease the number of threads used for classification
        fragmentSettingBinding.threadsMinus.setOnClickListener {
            if (numThreads > 1) {
                numThreads--
                updateDialogUi()
            }
        }

        // When clicked, increase the number of threads used for classification
        fragmentSettingBinding.threadsPlus.setOnClickListener {
            if (numThreads < 4) {
                numThreads++
                updateDialogUi()
            }
        }
    }

    //  Update the values displayed in the dialog.
    private fun updateDialogUi() {
        fragmentSettingBinding.threadsValue.text =
            numThreads.toString()
    }
}
