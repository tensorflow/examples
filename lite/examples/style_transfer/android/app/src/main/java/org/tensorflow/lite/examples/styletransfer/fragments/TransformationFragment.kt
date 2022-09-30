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

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.Toast
import androidx.core.content.ContextCompat
import androidx.fragment.app.activityViewModels
import androidx.recyclerview.widget.DividerItemDecoration
import androidx.recyclerview.widget.LinearLayoutManager
import com.bumptech.glide.Glide
import org.tensorflow.lite.examples.styletransfer.MainViewModel
import org.tensorflow.lite.examples.styletransfer.R
import org.tensorflow.lite.examples.styletransfer.StyleTransferHelper
import org.tensorflow.lite.examples.styletransfer.databinding.FragmentTransformationBinding
import java.io.InputStream

class TransformationFragment : Fragment(),
    StyleTransferHelper.StyleTransferListener {

    private var _fragmentTransformationBinding: FragmentTransformationBinding? =
        null
    private val fragmentTransformationBinding get() = _fragmentTransformationBinding!!
    private val viewModel: MainViewModel by activityViewModels()
    private lateinit var styleTransferHelper: StyleTransferHelper

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        // Inflate the layout for this fragment
        _fragmentTransformationBinding =
            FragmentTransformationBinding.inflate(inflater, container, false)

        return fragmentTransformationBinding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        styleTransferHelper = StyleTransferHelper(
            numThreads = viewModel.defaultModelNumThreads,
            currentDelegate = viewModel.defaultModelDelegate,
            currentModel = viewModel.defaultModel,
            context = requireContext(),
            styleTransferListener = this
        )

        // Set default value for controller view
        fragmentTransformationBinding.spinnerDelegate.setSelection(
            viewModel.defaultModelDelegate,
            false
        )
        fragmentTransformationBinding.spinnerModel.setSelection(
            viewModel.defaultModel,
            false
        )
        fragmentTransformationBinding.threadsValue.text =
            viewModel.defaultModelNumThreads.toString()


        // Setup list style image
        getListStyle().let { styles ->
            with(fragmentTransformationBinding.recyclerViewStyle) {
                val linearLayoutManager = LinearLayoutManager(
                    context,
                    LinearLayoutManager.HORIZONTAL, false
                )
                layoutManager = linearLayoutManager

                val dividerItemDecoration = DividerItemDecoration(
                    context,
                    linearLayoutManager.orientation
                )
                dividerItemDecoration.setDrawable(
                    ContextCompat.getDrawable
                        (context, R.drawable.decoration_divider)!!
                )
                addItemDecoration(dividerItemDecoration)
                adapter = StyleAdapter(styles) { pos ->
                    getBitmapFromAssets(
                        "thumbnails/${styles[pos].imagePath}"
                    )?.let {
                        styleTransferHelper.setStyleImage(it)
                    }
                }.apply {
                    // Set default style image
                    setSelected(0, true)
                    getBitmapFromAssets("thumbnails/${styles[0].imagePath}")?.let {
                        styleTransferHelper.setStyleImage(it)
                    }
                }
            }
        }

        // Attach listeners to UI control widgets
        initControllerViews()
        viewModel.getInputBitmap()?.let { originalBitmap ->
            // Display the original at the first time.
            Glide.with(requireActivity()).load(originalBitmap).centerCrop()
                .into(fragmentTransformationBinding.imgStyled)

            fragmentTransformationBinding.btnTransfer.setOnClickListener {
                styleTransferHelper.transfer(originalBitmap)
            }
        }
    }

    override fun onError(error: String) {
        activity?.runOnUiThread {
            Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
        }
    }

    override fun onResult(bitmap: Bitmap, inferenceTime: Long) {
        activity?.runOnUiThread {
            Glide.with(requireContext()).load(bitmap).centerCrop()
                .into(fragmentTransformationBinding.imgStyled)
            fragmentTransformationBinding.inferenceTimeVal.text =
                String.format("%d ms", inferenceTime)
        }
    }

    private fun initControllerViews() {
        // When clicked, decrease the number of threads used for transformation
        fragmentTransformationBinding.threadsMinus.setOnClickListener {
            if (styleTransferHelper.numThreads > 1) {
                styleTransferHelper.numThreads--
                updateControlsUi()
            }
        }

        // When clicked, increase the number of threads used for transformation
        fragmentTransformationBinding.threadsPlus.setOnClickListener {
            if (styleTransferHelper.numThreads < 4) {
                styleTransferHelper.numThreads++
                updateControlsUi()
            }
        }

        // When clicked, change the underlying hardware used for inference.
        // Current options are CPU, GPU, and NNAPI
        fragmentTransformationBinding.spinnerDelegate.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    parent: AdapterView<*>?,
                    view: View?,
                    position: Int,
                    id: Long
                ) {
                    styleTransferHelper.currentDelegate = position
                    viewModel.defaultModelDelegate = position
                    updateControlsUi()
                }

                override fun onNothingSelected(parent: AdapterView<*>?) {
                    /* no op */
                }
            }

        // When clicked, change the underlying model used for object
        // transformation
        fragmentTransformationBinding.spinnerModel.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    parent: AdapterView<*>?,
                    view: View?,
                    position: Int,
                    id: Long
                ) {
                    styleTransferHelper.currentModel = position
                    viewModel.defaultModel = position
                    updateControlsUi()
                }

                override fun onNothingSelected(parent: AdapterView<*>?) {
                    /* no op */
                }
            }
    }

    // Update the values displayed in the controller view. Reset transfer.
    private fun updateControlsUi() {
        viewModel.defaultModelNumThreads = styleTransferHelper.numThreads
        fragmentTransformationBinding.threadsValue.text =
            styleTransferHelper.numThreads.toString()
        // Needs to be cleared instead of reinitialized because the GPU
        // delegate needs to be initialized on the thread using it when applicable
        styleTransferHelper.clearStyleTransferHelper()
    }

    private fun getListStyle(): MutableList<StyleAdapter.Style> {
        val styles = mutableListOf<StyleAdapter.Style>()
        requireActivity().assets.list("thumbnails")?.forEach {
            styles.add(StyleAdapter.Style(it))
        }
        return styles
    }

    private fun getBitmapFromAssets(fileName: String): Bitmap? {
        val assetManager: AssetManager = requireActivity().assets
        return try {
            val istr: InputStream = assetManager.open(fileName)
            val bitmap = BitmapFactory.decodeStream(istr)
            istr.close()
            bitmap
        } catch (e: Exception) {
            null
        }
    }
}
