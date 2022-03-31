# The instrumentation tests use Tasks.await, but the instrumented code doesn't.
# This proguard config makes sure the method is not pruned at runtime.
-keepclasseswithmembers class com.google.android.gms.tasks.** { *; }