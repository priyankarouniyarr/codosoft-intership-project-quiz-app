import 'package:flutter/material.dart';
import 'package:quiz_app/models/models.dart';
import 'package:quiz_app/screen/resultpage.dart';

class Quizpage extends StatefulWidget {
  final QuizSet quizset;
  const Quizpage({super.key, required this.quizset});

  @override
  State<Quizpage> createState() => _QuizpageState();
}

class _QuizpageState extends State<Quizpage> {
  int currentquestionIndex = 0;
  List<int?> selectedIndexanswer = [];
  @override
  void initState() {
    super.initState();
    selectedIndexanswer =
        List<int?>.filled(widget.quizset.questions.length, null);
  }

  void nextQuestion() {
    setState(() {
      if (currentquestionIndex < widget.quizset.questions.length - 1) {
      currentquestionIndex++;
    }
    });
    
  }

  void backQuestion() {setState(() {
if (currentquestionIndex > 0) {
      currentquestionIndex--;
    }
  });
    
  }

  @override
  Widget build(BuildContext context) {
    final Question currentQuestion =
        widget.quizset.questions[currentquestionIndex];

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
                            )),
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
                          child: Text(currentQuestion.question,
                              style: TextStyle(
                                fontSize: 20,
                                color: Colors.white,
                                fontWeight: FontWeight.bold,
                              
                              ),textAlign: TextAlign.center,
                              ),
                        ),
                        SizedBox(
                          height: 10,
                        ),
                        ...currentQuestion.options.asMap().entries.map((entry) {
                          final index = entry.key;
                          final option = entry.value;
                          return GestureDetector(
                              onTap: () {
                                setState(() {
                                  selectedIndexanswer[currentquestionIndex] =
                                      index;
                                });
                              },
                              child: Container(
                                width: MediaQuery.of(context).size.width,
                                padding: EdgeInsets.symmetric(
                                    vertical: 15, horizontal: 5),
                                margin: EdgeInsets.all(5),
                                decoration: BoxDecoration(
                                  color: selectedIndexanswer[
                                              currentquestionIndex] ==
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
                                child: Text(option,
                                    style: TextStyle(
                                      fontSize: 18,
                                      color: selectedIndexanswer[
                                                  currentquestionIndex] ==
                                              index
                                          ? const Color.fromARGB(
                                              255, 39, 83, 160)
                                          : Colors.white,
                                      fontWeight: FontWeight.w800,
                                    ),textAlign: TextAlign.left,),
                              ));
                        }),
                        SizedBox(
                          height: 80,
                        ),
                        Padding(
                            padding: EdgeInsets.all(10),
                            child: Row(
                                mainAxisAlignment:
                                    MainAxisAlignment.spaceBetween,
                                children: [
                                  currentquestionIndex > 0 ? GestureDetector(
                                    onTap: 
                                     backQuestion,
                
                                    child: Container(
                                        width: 100,
                                        padding: EdgeInsets.all(10),
                                        decoration: BoxDecoration(
                                          color: Colors.blue,
                                          borderRadius:
                                              BorderRadius.circular(10),
                                        ),
                                        child: Center(
                                            child: Text("Previous",
                                                style: TextStyle(
                                                    fontSize: 18,
                                                    color: Colors.white,
                                                    fontWeight:
                                                        FontWeight.w500))))
                                  ):SizedBox(),

                                   GestureDetector(
                                    onTap: () {
                                      if (currentquestionIndex < widget. quizset.questions.length - 1) {
                                        nextQuestion();
                                    }
                                    else
                                    {
                                      int totalCorrect=0;
                                      for(int i=0;i<widget.quizset.questions.length;i++){
                                        if(selectedIndexanswer[i]==widget.quizset.questions[i].selectedIndex){
                                          totalCorrect++;
                                      }

                                    }
                                    Navigator.push(context,
                                    MaterialPageRoute(builder: (context) => ResultPage(
                                      totalAttempt: widget.quizset.questions.length,
                                      totalQuestion:   widget.quizset.questions.length,
                                      correctAnswer: totalCorrect,
                                      score: (totalCorrect*10) ,
                                      quizSet: widget.quizset,


                                    )));
                                     } ;},
                                    child: Container(
                                        width: 100,
                                        padding: EdgeInsets.all(10),
                                        decoration: BoxDecoration(
                                          color: Colors.blue,
                                          borderRadius:
                                              BorderRadius.circular(10),
                                        ),
                                        child: Center(
                                            child: Text(currentquestionIndex==widget.quizset.questions.length-1?"Submit":"Next",
                                                style: TextStyle(
                                                    fontSize: 18,
                                                    color: Colors.white,
                                                    fontWeight:
                                                        FontWeight.w500)))),
                                  )
                                  ,
                                ]))
                      ],
                    ),
                  ),
                ],
              ),
            )));
  }
}
