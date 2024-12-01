-keepclasseswithmembers class com.google.samples.gms.tflite.c.MainActivity { *; }
-keepclasseswithmembers class com.google.samples.gms.tflite.c.TfLiteJni { *; }
-keepclasseswithmembers class android.support.** { *; }
-keepclasseswithmembers class androidx.** { *; }

# The tests use Tasks.await, but the instrumented code doesn't
# Without the appropriate Proguard config, the method would be pruned
-keepclasseswithmembers class com.google.android.gms.tasks.** { *; }
# The tests also uses TfLiteNative.initialize(context)
-keepclasseswithmembers class com.google.android.gms.tflite.java.TfLiteNative { *; }
