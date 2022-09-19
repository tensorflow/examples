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

package org.tensorflow.lite.examples.digitclassification.fragments

import android.annotation.SuppressLint
import android.graphics.Color
import android.os.Bundle
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.DividerItemDecoration
import androidx.recyclerview.widget.LinearLayoutManager
import org.tensorflow.lite.examples.digitclassification.DigitClassifierHelper
import org.tensorflow.lite.examples.digitclassification.R
import org.tensorflow.lite.examples.digitclassification.databinding.FragmentDigitCanvasBinding
import org.tensorflow.lite.task.vision.classifier.Classifications
import java.util.Locale

class DigitCanvasFragment : Fragment(),
    DigitClassifierHelper.DigitClassifierListener {
    private var _fragmentDigitCanvasBinding: FragmentDigitCanvasBinding? = null
    private val fragmentDigitCanvasBinding get() = _fragmentDigitCanvasBinding!!
    private lateinit var digitClassifierHelper: DigitClassifierHelper
    private lateinit var classificationResultAdapter:
            ClassificationResultsAdapter

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _fragmentDigitCanvasBinding = FragmentDigitCanvasBinding.inflate(
            inflater,
            container, false
        )
        return fragmentDigitCanvasBinding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        digitClassifierHelper = DigitClassifierHelper(
            context = requireContext(), digitClassifierListener = this
        )
        setupDigitCanvas()
        setupClassificationResultAdapter()
        // Attach listeners to UI control widgets
        initBottomSheetControls()

        fragmentDigitCanvasBinding.btnClear.setOnClickListener {
            fragmentDigitCanvasBinding.digitCanvas.clearCanvas()
            classificationResultAdapter.reset()
        }
    }

    override fun onDestroyView() {
        _fragmentDigitCanvasBinding = null
        super.onDestroyView()
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun setupDigitCanvas() {
        with(fragmentDigitCanvasBinding.digitCanvas) {
            setStrokeWidth(70f)
            setColor(Color.WHITE)
            setBackgroundColor(Color.BLACK)
            setOnTouchListener { _, event ->
                // As we have interrupted DrawView's touch event, we first
                // need to pass touch events through to the instance for the drawing to show up
                onTouchEvent(event)

                // Then if user finished a touch event, run classification
                if (event.action == MotionEvent.ACTION_UP) {
                    classifyDrawing()
                }
                true
            }
        }
    }

    private fun setupClassificationResultAdapter() {
        classificationResultAdapter = ClassificationResultsAdapter()
        with(fragmentDigitCanvasBinding.recyclerViewResults) {
            adapter = classificationResultAdapter
            layoutManager = LinearLayoutManager(requireContext())
            addItemDecoration(
                DividerItemDecoration(
                    requireContext(),
                    DividerItemDecoration.VERTICAL
                )
            )
        }
    }

    private fun initBottomSheetControls() {
        // When clicked, lower classification score threshold floor
        fragmentDigitCanvasBinding.bottomSheetLayout.thresholdMinus.setOnClickListener {
            if (digitClassifierHelper.threshold >= 0.1) {
                digitClassifierHelper.threshold -= 0.1f
                updateControlsUi()
            }
        }

        // When clicked, raise classification score threshold floor
        fragmentDigitCanvasBinding.bottomSheetLayout.thresholdPlus.setOnClickListener {
            if (digitClassifierHelper.threshold < 0.9) {
                digitClassifierHelper.threshold += 0.1f
                updateControlsUi()
            }
        }

        // When clicked, reduce the number of objects that can be classified at a time
        fragmentDigitCanvasBinding.bottomSheetLayout.maxResultsMinus.setOnClickListener {
            if (digitClassifierHelper.maxResults > 1) {
                digitClassifierHelper.maxResults--
                updateControlsUi()
            }
        }

        // When clicked, increase the number of objects that can be classified at a time
        fragmentDigitCanvasBinding.bottomSheetLayout.maxResultsPlus.setOnClickListener {
            if (digitClassifierHelper.maxResults < 3) {
                digitClassifierHelper.maxResults++
                updateControlsUi()
            }
        }

        // When clicked, decrease the number of threads used for classification
        fragmentDigitCanvasBinding.bottomSheetLayout.threadsMinus.setOnClickListener {
            if (digitClassifierHelper.numThreads > 1) {
                digitClassifierHelper.numThreads--
                updateControlsUi()
            }
        }

        // When clicked, increase the number of threads used for classification
        fragmentDigitCanvasBinding.bottomSheetLayout.threadsPlus.setOnClickListener {
            if (digitClassifierHelper.numThreads < 4) {
                digitClassifierHelper.numThreads++
                updateControlsUi()
            }
        }

        // When clicked, change the underlying hardware used for inference. Current options are CPU
        // GPU, and NNAPI
        fragmentDigitCanvasBinding.bottomSheetLayout.spinnerDelegate.setSelection(
            0,
            false
        )
        fragmentDigitCanvasBinding.bottomSheetLayout.spinnerDelegate.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    parent: AdapterView<*>?,
                    view: View?,
                    position: Int,
                    id: Long
                ) {
                    digitClassifierHelper.currentDelegate = position
                    updateControlsUi()
                }

                override fun onNothingSelected(parent: AdapterView<*>?) {
                    /* no op */
                }
            }
    }

    // Update the values displayed in the bottom sheet. Reset classifier.
    private fun updateControlsUi() {
        fragmentDigitCanvasBinding.bottomSheetLayout.maxResultsValue.text =
            digitClassifierHelper.maxResults.toString()

        fragmentDigitCanvasBinding.bottomSheetLayout.thresholdValue.text =
            String.format(Locale.US, "%.2f", digitClassifierHelper.threshold)
        fragmentDigitCanvasBinding.bottomSheetLayout.threadsValue.text =
            digitClassifierHelper.numThreads.toString()
        // Needs to be cleared instead of reinitialized because the GPU
        // delegate needs to be initialized on the thread using it when applicable
        digitClassifierHelper.clearDigitClassifier()
    }

    private fun classifyDrawing() {
        val bitmap = fragmentDigitCanvasBinding.digitCanvas.getBitmap()
        digitClassifierHelper.classify(bitmap)
    }

    override fun onError(error: String) {
        activity?.runOnUiThread {
            Toast.makeText(requireActivity(), error, Toast.LENGTH_SHORT).show()
            classificationResultAdapter.reset()
        }
    }

    override fun onResults(
        results: List<Classifications>?,
        inferenceTime: Long
    ) {
        activity?.runOnUiThread {
            classificationResultAdapter.updateResults(results)
            fragmentDigitCanvasBinding.tvInferenceTime.text = requireActivity()
                .getString(R.string.inference_time, inferenceTime.toString())
        }
    }
}
