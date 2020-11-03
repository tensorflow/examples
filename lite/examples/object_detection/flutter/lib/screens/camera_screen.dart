import 'dart:async';

import 'package:tflite/tflite.dart';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';

const String ssd = "SSD MobileNet";
const String yolo = "Tiny YOLOv2";

List<CameraDescription> cameras;
typedef void Callback(List<dynamic> list, int h, int w);

class CameraScreen extends StatefulWidget {
  static const String id = 'camera_screen';

  @override
  _CameraScreenState createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController controller;
  List _recognitions;
  int _imageHeight;
  int _imageWidth;
  bool isDetecting = false;

  @override
  void initState() {
    super.initState();
    loadModel().then((val) {
      setState(() {});
    });

    controller = CameraController(cameras[0], ResolutionPreset.ultraHigh);
    controller.initialize().then((_) {
      if (!mounted) {
        return;
      }
      setState(() {});
      newFrame();
    });
  }

  // Takes a new frame
  void newFrame() {
    print('inside init');
    controller.startImageStream((CameraImage img) {
      if (!isDetecting) {
        isDetecting = true;
        print('loading ssd');
        ssdMobileNet(img);
        print('SSD completete');
      }
      setState(() {
        _imageHeight = img.width;
        _imageWidth = img.height;
      });
    });
  }

  @override
  void dispose() {
    controller?.dispose();
    super.dispose();
  }

  // loads the model with the required assets
  Future loadModel() async {
    Tflite.close();
    try {
      String res = await Tflite.loadModel(
        model: "assets/ssd_mobilenet.tflite",
        labels: "assets/ssd_mobilenet.txt",
      );

      print(res);
    } catch (e) {
      print('Failed to load model: $e');
    }
  }

// function to call the model: SSD
  Future ssdMobileNet(CameraImage img) async {
    int startTime = new DateTime.now().millisecondsSinceEpoch;
    print('inside SSD');
    var recognitions = await Tflite.detectObjectOnFrame(
        bytesList: img.planes.map((plane) {
          return plane.bytes;
        }).toList(), // required
        model: "SSDMobileNet",
        imageHeight: img.height,
        imageWidth: img.width,
        imageMean: 127.5, // defaults to 127.5
        imageStd: 127.5, // defaults to 127.5
        rotation: 90, // defaults to 90, Android only
        threshold: 0.1, // defaults to 0.1
        asynch: true // defaults to true
        );
    setState(() {
      _recognitions = recognitions;
      isDetecting = false;
    });
    int endTime = new DateTime.now().millisecondsSinceEpoch;
    print("Inference took ${endTime - startTime}ms");
  }

// renders boxes over our image along with its attributes
  List<Widget> renderBoxes(Size screen) {
    if (_recognitions == null) return [];
    if (_imageHeight == null || _imageWidth == null) return [];

    double factorX = screen.width;
    double factorY = _imageHeight / _imageWidth * screen.width;
    print('$factorX --- $factorY ----- $_imageHeight ------ $_imageWidth');
    Color blue = Color.fromRGBO(37, 213, 253, 1.0);
    List<dynamic> objects = [];
    for (var x in _recognitions) {
      if (x['confidenceInClass'] < 0.4) {
        break;
      }
      objects.add(x);
    }
    print(objects);
    return objects.map((re) {
      print(
          '${re["detectedClass"]} ${(re["confidenceInClass"] * 100).toStringAsFixed(0)}%');

      return Positioned(
        left: re["rect"]["x"] * factorX,
        top: re["rect"]["y"] * factorY,
        width: re["rect"]["w"] * factorX,
        height: re["rect"]["h"] * factorY,
        child: Container(
          decoration: BoxDecoration(
            borderRadius: BorderRadius.all(Radius.circular(8.0)),
            border: Border.all(
              color: blue,
              width: 2,
            ),
          ),
          child: Text(
            "${re["detectedClass"]} ${(re["confidenceInClass"] * 100).toStringAsFixed(0)}%",
            style: TextStyle(
              background: Paint()..color = blue,
              color: Colors.white,
              fontSize: 12.0,
            ),
          ),
        ),
      );
    }).toList();
  }

  @override
  Widget build(BuildContext context) {
    Size size = MediaQuery.of(context).size;
    List<Widget> stackChildren = [];
    print('width at stacking = ${size.width}');
    stackChildren.add(Positioned(
      top: 0.0,
      left: 0.0,
      width: size.width,
      child: !controller.value.isInitialized
          ? Padding(
              padding: const EdgeInsets.all(20.0),
              child: Text(
                'Camera \nsays \n404 \n:(',
                style: TextStyle(
                  color: Color(0xffaf8d6b),
                  fontSize: 40,
                  fontFamily: 'FjallaOne',
                ),
              ),
            )
          : AspectRatio(
              aspectRatio: controller.value.aspectRatio,
              child: CameraPreview(controller)),
    ));

    stackChildren.addAll(renderBoxes(size));

    return Scaffold(
      appBar: AppBar(
        title: Text(
          'TensorFlow Lite',
          style: TextStyle(
            fontFamily: 'FjallaOne',
          ),
        ),
        backgroundColor: Color(0xff1B7E81),
      ),
      body: Stack(
        children: stackChildren,
      ),
    );
  }
}
