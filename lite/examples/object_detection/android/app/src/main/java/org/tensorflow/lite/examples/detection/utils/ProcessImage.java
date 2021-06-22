package org.tensorflow.lite.examples.detection.utils;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.os.Environment;
import android.util.Log;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class ProcessImage {
    public static MappedByteBuffer loadModelFile(AssetManager assets, String modelFileName) throws IOException{
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFileName);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public static void softmax(final float[] vals){
        float max = Float.NEGATIVE_INFINITY;
        for(final float val : vals){
            max = Math.max(max, val);
        }
        float sum = 0.0f;
        for(int i = 0; i < vals.length; i++){
            vals[i] = (float) Math.exp(vals[i] - max);
            sum += vals[i];
        }
        for(int i = 0; i < vals.length; i++){
            vals[i] = vals[i] / sum;
        }
    }

    public static float expit(final float x){
        return (float) (1./(1. + Math.exp(-x)));
    }

    public static Matrix getTransformationMatrix(
            final int srcWidth,
            final int srcHeight,
            final int dstWidth,
            final int dstHeight,
            final int applyRotation,
            final boolean maintainAspectRatio
    ){
        final android.graphics.Matrix matrix = new android.graphics.Matrix();
        if(applyRotation != 0){
            matrix.postTranslate(-srcWidth / 2.0f, -srcHeight / 2.0f);
            matrix.postRotate(applyRotation);
        }

        final boolean transpose =(Math.abs(applyRotation) + 90)%180 == 0;
        final int inWidth = transpose ? srcHeight : srcWidth;
        final int inHeight = transpose ? srcWidth : srcHeight;

        if(inWidth != dstWidth || inHeight != dstHeight){
            final float scaleFactorx = dstWidth / (float) inWidth;
            final float scaleFactorY = dstHeight / (float) inHeight;

            if(maintainAspectRatio){
                final float scaleFactor = Math.max(scaleFactorx, scaleFactorY);
                matrix.postScale(scaleFactorx, scaleFactorY);
            }
        }

        if(applyRotation != 0){
            matrix.postTranslate(dstWidth / 2.0f, dstHeight / 2.0f);
        }
        return matrix;
    }

    public static Bitmap processBitmap(Bitmap source, int size){
        int image_height = source.getHeight();
        int image_width = source.getWidth();

        Bitmap croppedBitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888);
        Matrix frameToCropTransformations = getTransformationMatrix(image_width, image_height, size, size, 0, false);
        Matrix cropToFrameTransformations = new Matrix();
        frameToCropTransformations.invert(cropToFrameTransformations);

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(source, frameToCropTransformations, null);

        return croppedBitmap;
    }

    public static void writeToFile(String data, Context context){
        try {
            String baseDir = Environment.getExternalStorageDirectory().getAbsolutePath();
            String fileName = "myFile.txt";

            File file = new File(baseDir + File.separator + fileName);

            FileOutputStream stream = new FileOutputStream(file);
            try {
                stream.write(data.getBytes());
            }finally{
                stream.close();
            }
        }catch (IOException e){
            Log.e("Execption", "File write failed: " + e.toString());
        }
    }


}
