import 'dart:async';
import 'package:flutter/material.dart';
import 'package:quiz_app/models/models.dart';
import 'package:quiz_app/screen/homepage.dart';
import 'package:quiz_app/screen/resultpage.dart';

class Quizpage extends StatefulWidget {
  final QuizSet quizset;

  const Quizpage({super.key, required this.quizset});

  @override
  State<Quizpage> createState() => _QuizpageState();
}

class _QuizpageState extends State<Quizpage> {
  int seconds = 60;
  Timer? timer;
  int currentquestionIndex = 0;
  List<int?> selectedIndexanswer = [];

  @override
  void initState() {
    super.initState();
    starttimer();
    selectedIndexanswer =
        List<int?>.filled(widget.quizset.questions.length, null);
  }

  @override
 

  void starttimer() {
    timer = Timer.periodic(Duration(seconds: 1), (timer) {
      setState(() {
        if (seconds > 0) {
          seconds--;
        } else {
          timer.cancel();
          showTimeUpScreen();
        }
      });
    });
  }

   void dispose() {
    timer?.cancel();
    super.dispose();
  }


  void showTimeUpScreen() {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => AlertDialog(
        backgroundColor: const Color.fromARGB(255, 46, 143, 138),
        title: Text(
          'Time Up!',
          style: TextStyle(color: Colors.white),
        ),
        content: Text(
          'The time for this quiz has ended.',
          style: TextStyle(color: Colors.white),
        ),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.of(context).pop();
              Navigator.of(context).pushReplacement(
                MaterialPageRoute(builder: (context) => Homepage()),
              );
            },
            child: Container(
              padding: EdgeInsets.symmetric(vertical: 10, horizontal: 20),
              decoration: BoxDecoration(
                color: Colors.blue,
                borderRadius: BorderRadius.circular(5),
              ),
              child: Text(
                'Restart',
                style: TextStyle(
                  color: Colors.white,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  void nextQuestion() {
    setState(() {
      if (currentquestionIndex < widget.quizset.questions.length - 1) {
        currentquestionIndex++;
      }
    });
  }

  void backQuestion() {
    setState(() {
      if (currentquestionIndex > 0) {
        currentquestionIndex--;
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final Question currentQuestion =
        widget.quizset.questions[currentquestionIndex];

    bool isAnswerSelected =
        selectedIndexanswer[currentquestionIndex] != null;

    bool isquestioneditable = currentquestionIndex == selectedIndexanswer.length - 1 ||
        selectedIndexanswer[currentquestionIndex] == null;

    return Scaffold(
      body: Container(
        height: double.infinity,
        width: MediaQuery.of(context).size.width,
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [
              Color.fromARGB(255, 70, 97, 218),
              Color.fromARGB(255, 214, 120, 156),
              Color.fromARGB(255, 109, 102, 209)
            ],
            begin: Alignment.topLeft,
            end: Alignment.bottomCenter,
          ),
        ),
        child: SingleChildScrollView(
          physics: NeverScrollableScrollPhysics(),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              SizedBox(height: 40),
              Padding(
                padding: EdgeInsets.symmetric(horizontal: 15.0),
                child: Row(
                  children: [
                    GestureDetector(
                      onTap: () {
                        Navigator.pop(context);
                      },
                      child: Icon(
                        Icons.arrow_back_ios,
                        color: Colors.white,
                        size: 30,
                      ),
                    ),
                    Text(
                      widget.quizset.name,
                      style: TextStyle(
                        fontSize: 20,
                        color: Colors.white,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ],
                ),
              ),
              Stack(
                alignment: Alignment.center,
                children: [
                  Center(
                    child: Container(
                      width: 80,
                      height: 50,
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.5),
                        shape: BoxShape.circle,
                      ),
                      alignment: Alignment.center,
                      child: Text(
                        "$seconds",
                        style: TextStyle(
                          color: seconds <= 10 ? Colors.red : Colors.black,
                          fontSize: 22,
                        ),
                      ),
                    ),
                  ),
                  Center(
                    child: SizedBox(
                      width: 100,
                      height: 90,
                      child: CircularProgressIndicator(
                        value: seconds / 60,
                        valueColor: AlwaysStoppedAnimation(
                          seconds <= 10 ? Colors.red : Colors.lightGreen,
                        ),
                        backgroundColor: Colors.grey.withOpacity(0.3),
                        strokeWidth: 10,
                      ),
                    ),
                  ),
                ],
              ),
              Container(
                width: MediaQuery.of(context).size.width,
                height: MediaQuery.of(context).size.height / 1,
                padding: EdgeInsets.all(16),
                margin: EdgeInsets.all(16),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.only(
                      topRight: Radius.circular(10),
                      topLeft: Radius.circular(10)),
                  color: Colors.black.withOpacity(0.5),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Center(
                      child: Text(
                        currentQuestion.question,
                        style: TextStyle(
                          fontSize: 20,
                          color: Colors.white,
                          fontWeight: FontWeight.bold,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ),
                    SizedBox(
                      height: 10,
                    ),
                    ...currentQuestion.options.asMap().entries.map((entry) {
                      final index = entry.key;
                      final option = entry.value;
                      return GestureDetector(
                        onTap: isquestioneditable
                            ? () {
                                setState(() {
                                  selectedIndexanswer[currentquestionIndex] =
                                      index;
                                });
                              }
                            : null,
                        child: Container(
                          width: MediaQuery.of(context).size.width,
                          padding: EdgeInsets.symmetric(
                              vertical: 15, horizontal: 5),
                          margin: EdgeInsets.all(5),
                          decoration: BoxDecoration(
                            color: selectedIndexanswer[currentquestionIndex] ==
                                    index
                                ? const Color.fromARGB(255, 163, 204, 204)
                                : Colors.black.withOpacity(0.6),
                            borderRadius: BorderRadius.circular(10),
                            border: Border.all(
                                color: selectedIndexanswer[
                                            currentquestionIndex] ==
                                        index
                                    ? Colors.indigoAccent
                                    : Colors.grey,
                                width: 2),
                          ),
                          child: Text(
                            option,
                            style: TextStyle(
                              fontSize: 18,
                              color: selectedIndexanswer[
                                          currentquestionIndex] ==
                                      index
                                  ? const Color.fromARGB(255, 39, 83, 160)
                                  : Colors.white,
                              fontWeight: FontWeight.w800,
                            ),
                            textAlign: TextAlign.left,
                          ),
                        ),
                      );
                    }),
                    SizedBox(
                      height: 80,
                    ),
                    Padding(
                      padding: EdgeInsets.all(10),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          currentquestionIndex > 0
                              ? GestureDetector(
                                  onTap: backQuestion,
                                  child: Container(
                                    width: 100,
                                    padding: EdgeInsets.all(10),
                                    decoration: BoxDecoration(
                                      color: Colors.blue,
                                      borderRadius: BorderRadius.circular(10),
                                    ),
                                    child: Center(
                                      child: Text(
                                        "Previous",
                                        style: TextStyle(
                                          fontSize: 18,
                                          color: Colors.white,
                                          fontWeight: FontWeight.w500,
                                        ),
                                      ),
                                    ),
                                  ),
                                )
                              : SizedBox(),
                          GestureDetector(
                            onTap: isAnswerSelected
                                ? () {
                                      
                                    if (currentquestionIndex <
                                        widget.quizset.questions.length - 1) {
                                      nextQuestion();
                                    } else {
                                      int totalCorrect = 0;
                                      for (int i = 0;
                                          i < widget.quizset.questions.length;
                                          i++) {
                                        if (selectedIndexanswer[i] ==
                                            widget.quizset.questions[i]
                                                .selectedIndex) {
                                          totalCorrect++;
                                        }
                                      }
                                      Navigator.push(
                                        context,
                                        MaterialPageRoute(
                                          builder: (context) => ResultPage(
                                            totalAttempt: widget
                                                .quizset.questions.length,
                                            totalQuestion: widget
                                                .quizset.questions.length,
                                            correctAnswer: totalCorrect,
                                            score: (totalCorrect /
                                                    widget.quizset.questions
                                                        .length) *
                                                100,
                                            quizSet:
                                                widget.quizset,
                                          ),
                                        ),
                                      );
                                    }
                                  }
                                : null, 
                            child: Container(
                              width: 100,
                              padding: EdgeInsets.all(10),
                              decoration: BoxDecoration(
                                color: isAnswerSelected
                                    ? Colors.blue
                                    : Colors.grey, 
                                borderRadius: BorderRadius.circular(10),
                              ),
                              child: Center(
                                child: Text(
                                  currentquestionIndex ==
                                          widget.quizset.questions.length - 1
                                      ? "Submit"
                                      : "Next",
                                  style: TextStyle(
                                    fontSize: 18,
                                    color: Colors.white,
                                    fontWeight: FontWeight.w500,
                                  ),
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
