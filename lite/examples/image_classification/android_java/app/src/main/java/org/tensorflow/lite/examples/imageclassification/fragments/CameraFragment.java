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
package org.tensorflow.lite.examples.imageclassification.fragments;

import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.navigation.Navigation;
import androidx.recyclerview.widget.LinearLayoutManager;
import com.google.common.util.concurrent.ListenableFuture;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.tensorflow.lite.examples.imageclassification.ImageClassifierHelper;
import org.tensorflow.lite.examples.imageclassification.R;
import org.tensorflow.lite.examples.imageclassification.databinding.FragmentCameraBinding;
import org.tensorflow.lite.task.vision.classifier.Classifications;

/** Fragment for displaying and controlling the device camera and other UI */
public class CameraFragment extends Fragment
        implements ImageClassifierHelper.ClassifierListener {
    private static final String TAG = "Image Classifier";

    private FragmentCameraBinding fragmentCameraBinding;
    private ImageClassifierHelper imageClassifierHelper;
    private Bitmap bitmapBuffer;
    private ClassificationResultAdapter classificationResultsAdapter;
    private ImageAnalysis imageAnalyzer;
    private ProcessCameraProvider cameraProvider;
    private final Object task = new Object();

    /**
     * Blocking camera operations are performed using this executor
     */
    private ExecutorService cameraExecutor;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater,
                             @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        fragmentCameraBinding = FragmentCameraBinding
                .inflate(inflater, container, false);
        return fragmentCameraBinding.getRoot();
    }

    @Override
    public void onResume() {
        super.onResume();

        if (!PermissionsFragment.hasPermission(requireContext())) {
            Navigation.findNavController(requireActivity(), R.id.fragment_container)
                    .navigate(
                            CameraFragmentDirections.actionCameraToPermissions()
                    );
        }
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();

        // Shut down our background executor
        cameraExecutor.shutdown();
        synchronized (task) {
            imageClassifierHelper.clearImageClassifier();
        }
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        cameraExecutor = Executors.newSingleThreadExecutor();
        imageClassifierHelper = ImageClassifierHelper.create(requireContext()
                , this);

        // setup result adapter
        classificationResultsAdapter = new ClassificationResultAdapter();
        classificationResultsAdapter
                .updateAdapterSize(imageClassifierHelper.getMaxResults());
        fragmentCameraBinding.recyclerviewResults
                .setAdapter(classificationResultsAdapter);
        fragmentCameraBinding.recyclerviewResults
                .setLayoutManager(new LinearLayoutManager(requireContext()));

        // Set up the camera and its use cases
        fragmentCameraBinding.viewFinder.post(this::setUpCamera);

        // Attach listeners to UI control widgets
        initBottomSheetControls();
    }

    @Override
    public void onConfigurationChanged(@NonNull Configuration newConfig) {
        super.onConfigurationChanged(newConfig);
        imageAnalyzer.setTargetRotation(
                fragmentCameraBinding.viewFinder.getDisplay().getRotation()
        );
    }

    private void initBottomSheetControls() {
        // When clicked, lower classification score threshold floor
        fragmentCameraBinding.bottomSheetLayout.thresholdMinus
                .setOnClickListener(view -> {
                    float threshold = imageClassifierHelper.getThreshold();
                    if (threshold >= 0.1) {
                        imageClassifierHelper.setThreshold(threshold - 0.1f);
                        updateControlsUi();
                    }
                });

        // When clicked, raise classification score threshold floor
        fragmentCameraBinding.bottomSheetLayout.thresholdPlus
                .setOnClickListener(view -> {
                    float threshold = imageClassifierHelper.getThreshold();
                    if (threshold < 0.9) {
                        imageClassifierHelper.setThreshold(threshold + 0.1f);
                        updateControlsUi();
                    }
                });

        // When clicked, reduce the number of objects that can be classified
        // at a time
        fragmentCameraBinding.bottomSheetLayout.maxResultsMinus
                .setOnClickListener(view -> {
                    int maxResults = imageClassifierHelper.getMaxResults();
                    if (maxResults > 1) {
                        imageClassifierHelper.setMaxResults(maxResults - 1);
                        updateControlsUi();
                        classificationResultsAdapter.updateAdapterSize(
                                imageClassifierHelper.getMaxResults()
                        );
                    }
                });

        // When clicked, increase the number of objects that can be
        // classified at a time
        fragmentCameraBinding.bottomSheetLayout.maxResultsPlus
                .setOnClickListener(view -> {
                    int maxResults = imageClassifierHelper.getMaxResults();
                    if (maxResults < 3) {
                        imageClassifierHelper.setMaxResults(maxResults + 1);
                        updateControlsUi();
                        classificationResultsAdapter.updateAdapterSize(
                                imageClassifierHelper.getMaxResults()
                        );
                    }
                });

        // When clicked, decrease the number of threads used for classification
        fragmentCameraBinding.bottomSheetLayout.threadsMinus
                .setOnClickListener(view -> {
                    int numThreads = imageClassifierHelper.getNumThreads();
                    if (numThreads > 1) {
                        imageClassifierHelper.setNumThreads(numThreads - 1);
                        updateControlsUi();
                    }
                });

        // When clicked, increase the number of threads used for classification
        fragmentCameraBinding.bottomSheetLayout.threadsPlus
                .setOnClickListener(view -> {
                    int numThreads = imageClassifierHelper.getNumThreads();
                    if (numThreads < 4) {
                        imageClassifierHelper.setNumThreads(numThreads + 1);
                        updateControlsUi();
                    }
                });

        // When clicked, change the underlying hardware used for inference.
        // Current options are CPU,GPU, and NNAPI
        fragmentCameraBinding.bottomSheetLayout.spinnerDelegate
                .setSelection(0, false);
        fragmentCameraBinding.bottomSheetLayout.spinnerDelegate
                .setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
                    @Override
                    public void onItemSelected(AdapterView<?> adapterView,
                                               View view,
                                               int position,
                                               long id) {
                        imageClassifierHelper.setCurrentDelegate(position);
                        updateControlsUi();
                    }

                    @Override
                    public void onNothingSelected(AdapterView<?> adapterView) {
                        // no-op
                    }
                });

        // When clicked, change the underlying model used for object
        // classification
        fragmentCameraBinding.bottomSheetLayout.spinnerModel
                .setSelection(0, false);
        fragmentCameraBinding.bottomSheetLayout.spinnerModel
                .setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
                    @Override
                    public void onItemSelected(AdapterView<?> adapterView,
                                               View view,
                                               int position,
                                               long id) {
                        imageClassifierHelper.setCurrentModel(position);
                        updateControlsUi();
                    }

                    @Override
                    public void onNothingSelected(AdapterView<?> adapterView) {
                        // no-op
                    }
                });
    }

    // Update the values displayed in the bottom sheet. Reset classifier.
    private void updateControlsUi() {
        fragmentCameraBinding.bottomSheetLayout.maxResultsValue
                .setText(String.valueOf(imageClassifierHelper.getMaxResults()));
        fragmentCameraBinding.bottomSheetLayout.thresholdValue
                .setText(String.format(Locale.US, "%.2f",
                        imageClassifierHelper.getThreshold()));
        fragmentCameraBinding.bottomSheetLayout.threadsValue
                .setText(String.valueOf(imageClassifierHelper.getNumThreads()));
        // Needs to be cleared instead of reinitialized because the GPU
        // delegate needs to be initialized on the thread using it when
        // applicable
        synchronized (task) {
            imageClassifierHelper.clearImageClassifier();
        }
    }

    // Initialize CameraX, and prepare to bind the camera use cases
    private void setUpCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(requireContext());
        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();

                // Build and bind the camera use cases
                bindCameraUseCases();
            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(requireContext()));
    }

    // Declare and bind preview, capture and analysis use cases
    private void bindCameraUseCases() {
        // CameraSelector - makes assumption that we're only using the back
        // camera
        CameraSelector.Builder cameraSelectorBuilder = new CameraSelector.Builder();
        CameraSelector cameraSelector = cameraSelectorBuilder
                .requireLensFacing(CameraSelector.LENS_FACING_BACK).build();

        // Preview. Only using the 4:3 ratio because this is the closest to
        // our model
        Preview preview = new Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(
                        fragmentCameraBinding.viewFinder
                                .getDisplay().getRotation()
                )
                .build();

        // ImageAnalysis. Using RGBA 8888 to match how our models work
        imageAnalyzer = new ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(fragmentCameraBinding.viewFinder.getDisplay().getRotation())
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build();

        // The analyzer can then be assigned to the instance
        imageAnalyzer.setAnalyzer(cameraExecutor, image -> {
            if (bitmapBuffer == null) {
                bitmapBuffer = Bitmap.createBitmap(
                        image.getWidth(),
                        image.getHeight(),
                        Bitmap.Config.ARGB_8888);
            }
            classifyImage(image);
        });

        // Must unbind the use-cases before rebinding them
        cameraProvider.unbindAll();

        try {
            // A variable number of use-cases can be passed here -
            // camera provides access to CameraControl & CameraInfo
            cameraProvider.bindToLifecycle(
                    this,
                    cameraSelector,
                    preview,
                    imageAnalyzer
            );

            // Attach the viewfinder's surface provider to preview use case
            preview.setSurfaceProvider(
                    fragmentCameraBinding.viewFinder.getSurfaceProvider()
            );
        } catch (Exception exc) {
            Log.e(TAG, "Use case binding failed", exc);
        }
    }

    private void classifyImage(@NonNull ImageProxy image) {
        // Copy out RGB bits to the shared bitmap buffer
        bitmapBuffer.copyPixelsFromBuffer(image.getPlanes()[0].getBuffer());

        int imageRotation = image.getImageInfo().getRotationDegrees();
        image.close();
        synchronized (task) {
            // Pass Bitmap and rotation to the image classifier helper for
            // processing and classification
            imageClassifierHelper.classify(bitmapBuffer, imageRotation);
        }
    }

    @Override
    public void onError(String error) {
        requireActivity().runOnUiThread(() -> {
            Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show();
            classificationResultsAdapter.updateResults(new ArrayList<>());
        });
    }

    @Override
    public void onResults(List<Classifications> results, long inferenceTime) {
        requireActivity().runOnUiThread(() -> {
            classificationResultsAdapter.updateResults(results.get(0).getCategories());
            fragmentCameraBinding.bottomSheetLayout.inferenceTimeVal
                    .setText(String.format(Locale.US, "%d ms", inferenceTime));
        });
    }
}
