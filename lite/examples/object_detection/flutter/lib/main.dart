import 'package:TfLite/screens/camera_screen.dart';
import 'package:TfLite/screens/home_screen.dart';
import 'package:flutter/material.dart';
import 'package:TfLite/screens/all_models.dart';
import 'package:camera/camera.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData.light().copyWith(
          floatingActionButtonTheme: FloatingActionButtonThemeData(
        backgroundColor: Color(0xffA0766E),
      )),
      initialRoute: HomeScreen.id,
      routes: {
        HomeScreen.id: (context) => HomeScreen(),
        CameraScreen.id: (context) => CameraScreen(),
        AllModels.id: (context) => AllModels(),
      },
    );
  }
}
