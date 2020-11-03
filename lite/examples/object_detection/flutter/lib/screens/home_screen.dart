import 'package:TfLite/screens/all_models.dart';
import 'package:TfLite/screens/camera_screen.dart';
import 'package:flutter/material.dart';
import 'package:TfLite/components/roundbutton.dart';

class HomeScreen extends StatelessWidget {
  static const String id = 'home_screen';
  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Scaffold(
        backgroundColor: Color(0xff1B7E81),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.center,
            children: <Widget>[
              RoundButton(
                title: 'Real-Time Detection',
                text:
                    'This feature allows you to turn on your camera and detect any objetcs that you point with your camera.',
                onPressed: () {
                  Navigator.pushNamed(context, CameraScreen.id);
                },
              ),
              SizedBox(
                height: 20,
              ),
              RoundButton(
                title: 'Classic Detection',
                text:
                    'In this feature you can either upload an image from your local storage or click a photo with your camera to detect any objects.',
                onPressed: () {
                  Navigator.pushNamed(context, AllModels.id);
                },
              ),
            ],
          ),
        ),
      ),
    );
  }
}
