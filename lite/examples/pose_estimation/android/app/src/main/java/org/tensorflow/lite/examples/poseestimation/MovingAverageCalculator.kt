package org.tensorflow.lite.examples.poseestimation

import java.lang.IllegalArgumentException
import java.util.LinkedList
import java.util.Queue

class MovingAverageCalculator(private val windowSize:Int) {
    private var activeValueQueue: Queue<Float> = LinkedList()
    private var sum: Float = 0f

    init {
        if (windowSize <= 0) {
            throw IllegalArgumentException("Window size must be positive.")
        }
    }

    public fun add(newItem: Float) {
        if (activeValueQueue.size >= windowSize) {
            val oldestItem = activeValueQueue.remove()
            sum -= oldestItem
        }
        sum += newItem
        activeValueQueue.add(newItem)
    }

    public fun average(): Float {
        return if (activeValueQueue.size > 0) (sum / activeValueQueue.size) else 0f
    }
}