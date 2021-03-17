package org.tensorflow.lite.examples.soundclassifier;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.util.Log;

import java.nio.FloatBuffer;

public class AudioBuffer {

    // TODO: What if the ring buffer is not yet fully filled before invocation?
    private class FloatRingBuffer {
        private float[] buffer;
        private int current;

        FloatRingBuffer(int size) {
            buffer = new float[size];
        }

        public int getCapacity() {
            return buffer.length;
        }

        public void feed(float v) {
            buffer[current] = v;
            current = (current + 1) % buffer.length;
        }

        public void feed(float[] data, int size) {
            for (int i = 0; i < size; i++) {
                feed(data[i]);
            }
        }

        public float[] getArray() {
            float[] output = new float[buffer.length];
            for (int i = current; i < buffer.length; i++) {
                output[i - current] = buffer[i];
            }
            for (int i = 0; i < current; i++) {
                output[buffer.length - current + i] = buffer[i];
            }
            return output;
        }
    }

    private AudioFormat audioFormat;
    // Keeping all the data in float and convert them to desired types at the end.
    // In the future, we might want to store data in the format specified by audioFormat.
    private FloatRingBuffer floatRingBuffer;

    // TODO: Do we need to keep track of the AudioFormat?
    public AudioBuffer(AudioFormat audioFormat, int sampleCount) {
        this.audioFormat = audioFormat;
        this.floatRingBuffer = new FloatRingBuffer(sampleCount);
    }

    // PCM float
    public int feed(float[] data) {
        return feed(data, data.length);
    }

    // TODO: what's the correct name?
    public int feed(float[] data, int size) {
        floatRingBuffer.feed(data, size);
        return size;
    }

    // PCM int16
    public int feed(short[] data) {
        return feed(data, data.length);
    }

    private float pcm16ToFloat(short v) {
        return (float) v / 32768;
    }

    public int feed(short[] data, int size) {
        for (int i = 0; i < size; i++) {
            floatRingBuffer.feed(pcm16ToFloat(data[i]));
        }
        return size;
    }

    // Read from AudioRecord as a helper function
    public int feed(AudioRecord record) {
        return feed(record, floatRingBuffer.getCapacity());
    }

    private int feed(AudioRecord record, float[] temporary) {
        int readSamples = record.read(temporary, 0, temporary.length, AudioRecord.READ_BLOCKING);
        if (readSamples > 0) {
            feed(temporary, readSamples);
        }
        return readSamples;
    }

    public int feed(AudioRecord record, short[] temporary) {
        int readSamples = record.read(temporary, 0, temporary.length, AudioRecord.READ_BLOCKING);
        if (readSamples > 0) {
            feed(temporary, readSamples);
        }
        return readSamples;
    }

    public int feed(AudioRecord record, int size) {
//        assert record.getChannelCount() == 1;
//        assert record.getSampleRate() == this.audioFormat.getSampleRate();

        int readSamples = 0;
        switch (record.getAudioFormat()) {
            case AudioFormat.ENCODING_PCM_FLOAT:
                readSamples = feed(record, new float[size]);
                break;
            case AudioFormat.ENCODING_PCM_16BIT:
                readSamples = feed(record, new short[size]);
                break;
            default:
                Log.e(TAG, "Unsupported AudioFormat. Requires either PCM float or PCM 16.");
        }

        // Report errors.
        switch (readSamples) {
            case AudioRecord.ERROR_INVALID_OPERATION:
                Log.w(TAG, "AudioRecord.ERROR_INVALID_OPERATION");
                break;
            case AudioRecord.ERROR_BAD_VALUE:
                Log.w(TAG, "AudioRecord.ERROR_BAD_VALUE");
                break;
            case AudioRecord.ERROR_DEAD_OBJECT:
                Log.w(TAG, "AudioRecord.ERROR_DEAD_OBJECT");
                break;
            case AudioRecord.ERROR:
                Log.w(TAG, "AudioRecord.ERROR");
                break;
        }
        return readSamples;

    }

    public AudioFormat getAudioFormat() {
        return audioFormat;
    }

    // TODO: ownership
    public FloatBuffer GetAudioBufferInFloat() {
        FloatBuffer output = FloatBuffer.wrap(this.floatRingBuffer.getArray());
        return output;
    }

    private String TAG = "AudioBuffer";
}
