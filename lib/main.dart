import 'package:flutter/material.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

final player = AudioPlayer()..setReleaseMode(ReleaseMode.loop);

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  var response;
  final TextEditingController inputController = TextEditingController();
  String outputResult = '';
  String url = '';

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Flutter App'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            TextField(
              controller: inputController,
              onChanged: (value) {
                url = 'http://10.0.2.2:8080/predict?q=$value';
              },
              decoration: InputDecoration(
                hintText: 'Enter a sentence',
              ),
            ),
            SizedBox(height: 16.0),
            ElevatedButton(
              onPressed: () async {
                String result = await fetchData(url);
                setState(() {
                  outputResult = result;
                });
              },
              child: Text('Make API Request'),
            ),
            SizedBox(height: 16.0),
            Text(
              'Output:\n$outputResult',
              style: TextStyle(fontSize: 18.0),
            ),
          ],
        ),
      ),
    );
  }

  fetchData(String url) async {
    http.Response response = await http.get(Uri.parse(url));
    if (response.body == '1')
      return '錯誤';
    else
      return '正確';
  }
/*Future<String> fetchData(String input) async {
    try {
      response = await fetchData(url);

      if (response.statusCode == 200) {
        return response.body;
      } else {
        throw Exception('Failed to load data: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Failed to load data: $e');
    }
  }*/
}
