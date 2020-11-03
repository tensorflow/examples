import 'package:flutter/material.dart';

class RoundButton extends StatelessWidget {
  RoundButton(
      {@required this.title, @required this.onPressed, @required this.text});

  final String title;
  final String text;
  final Function onPressed;

  @override
  Widget build(BuildContext context) {
    return Container(
      height: MediaQuery.of(context).size.width * 0.65,
      width: MediaQuery.of(context).size.width * 0.90,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.all(
          Radius.circular(20),
        ),
        color: Color(0xffffffff),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 12, 0, 4),
            child: Text(
              title,
              style: TextStyle(
                color: Color(0xff1B7E81),
                fontSize: 22,
                fontWeight: FontWeight.w500,
                fontFamily: 'FjallaOne',
              ),
            ),
          ),
          Divider(
            indent: 12,
            endIndent: 12,
            thickness: 2,
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 5),
            child: Text(
              text,
              style: TextStyle(
                fontSize: 14,
                color: Color(0xffa58d6b),
                fontFamily: 'FjallaOne',
              ),
            ),
          ),
          Expanded(
            child: Padding(
              padding: const EdgeInsets.all(12.0),
              child: Align(
                alignment: Alignment.bottomRight,
                child: RaisedButton.icon(
                  label: Text(
                    'Try Now',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 15,
                      fontFamily: 'FjallaOne',
                    ),
                  ),
                  icon: Icon(
                    Icons.keyboard_arrow_right,
                    size: 25,
                    color: Colors.white,
                  ),
                  onPressed: onPressed,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(5.0),
                  ),
                  color: Color(0xffA0766E),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
