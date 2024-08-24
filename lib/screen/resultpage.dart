import 'package:flutter/material.dart';
import 'package:quiz_app/models/models.dart';
import 'package:quiz_app/screen/homepage.dart';
import 'package:quiz_app/screen/quizpage.dart';

class ResultPage extends StatelessWidget {
  final int totalQuestion;
  final int score;
  final int correctAnswer;
  final QuizSet quizSet;
  final int totalAttempt;

  const ResultPage({
    super.key,
    required this.totalQuestion,
    required this.score,
    required this.correctAnswer,
    required this.quizSet,
    required this.totalAttempt,
  });

  @override
  Widget build(BuildContext context) {
    
    String result;
    if (score < 25) {
      result = "Oops! You Failed! \n   Your Score \n      $score/$totalQuestion";
    } else if (score < 50) {
      result = "Keep it up! \n   Your Score \n     $score/$totalQuestion";
    } else if (score < 80) {
      result = "Excellent! \n   Your Score \n   $score/$totalQuestion";
    } else {
      result = "Well done! \n   Your Score \n   $score/$totalQuestion";
    }

    return Scaffold(
      body: Container(
        height: double.infinity,
        width: MediaQuery.of(context).size.width,
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [
              Color.fromARGB(255, 70, 97, 218),
              Color.fromARGB(255, 214, 120, 156),
              Color.fromARGB(255, 109, 102, 209),
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
                        size: 25,
                      ),
                    ),
                    Flexible(
                      child: Text(
                        "Quiz Score",
                        style: TextStyle(
                          fontSize: 20,
                          color: Colors.white,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            SizedBox(
              height: MediaQuery.of(context).size.height/6,
            ),
              Container(
                height:MediaQuery.of(context).size.height,
                width: MediaQuery.of(context).size.width,
                padding: EdgeInsets.all(25),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.only(
                    topRight: Radius.circular(15),
                    topLeft: Radius.circular(15),
                  ),
                  color: Colors.black.withOpacity(0.5),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    Padding(
                      padding: const EdgeInsets.only(top:50.0),
                      child: Center(
                        child: Text(
                          result,
                          style: TextStyle(
                            fontSize: 30,
                            fontWeight: FontWeight.w600,
                            color: Colors.white,
                          ),
                        ),
                      ),
                    ),
                    SizedBox(height: 100,),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                  
                      children: [
                          GestureDetector(
                        onTap: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(builder: (context) => Homepage()),);},
                            child: Container(
                              padding:EdgeInsets.symmetric(vertical: 10,horizontal: 20),
                              decoration: BoxDecoration(
                                 gradient: LinearGradient(
                          colors: [
                            Colors.blueAccent,
                            Color.fromARGB(255, 0, 183, 255),
                          ],
                          begin: Alignment.topLeft,
                          end: Alignment.bottomRight,
                        ),
                        borderRadius: BorderRadius.circular(20),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.blueAccent.withOpacity(0.4),
                            blurRadius: 10,
                            offset: Offset(0, 5),
                          ),
                        ],

                              ),
                              child:Center(
                                child: Text('Cancel',style:TextStyle(
                                  fontSize: 18,
                                  color: Colors.white,
                                  fontWeight: FontWeight.w500,
                                )),
                              )

                            )
                      ),

                      SizedBox(
                        width: 100,
                      ),
                      GestureDetector(
                        onTap: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(builder: (context) => Quizpage(quizset: quizSet,)),);},
                            child: Container(
                              padding:EdgeInsets.symmetric(vertical: 10,horizontal: 20),
                              decoration: BoxDecoration(
                                 gradient: LinearGradient(
                          colors: [
                            Colors.blueAccent,
                            Color.fromARGB(255, 0, 183, 255),
                          ],
                          begin: Alignment.topLeft,
                          end: Alignment.bottomRight,
                        ),
                        borderRadius: BorderRadius.circular(20),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.blueAccent.withOpacity(0.4),
                            blurRadius: 10,
                            offset: Offset(0, 5),
                          ),
                        ],

                              ),
                              child:Center(
                                child: Text('Restart',style:TextStyle(
                                  fontSize: 18,
                                  color: Colors.white,
                                  fontWeight: FontWeight.w500,
                                )),
                              )

                            )
                      )
                    ],)
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
