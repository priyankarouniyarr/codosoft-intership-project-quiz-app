import 'package:flutter/material.dart';
import 'package:quiz_app/data/data.dart';
import 'package:quiz_app/models/models.dart';
import 'package:quiz_app/screen/quizpage.dart';

class CategoryPage extends StatelessWidget {
  final Category category;
  const CategoryPage({super.key,required this.category});
    @override
  Widget build(BuildContext context) {
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
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              SizedBox(height:40 ),
              Padding(
                padding: EdgeInsets.symmetric(horizontal: 15.0),
                child: Row(
                  children: [
                    GestureDetector(
                      onTap: () {
                        Navigator.pop(context);
                      },
                      child:Icon(Icons.arrow_back_ios,
                      color: Colors.white,
                      size:25,
                      )
                      ),
                       Flexible(
                         child: Text(
                          '${category.name} Quiz',
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
         ListView.builder(
           shrinkWrap:true,
           physics: NeverScrollableScrollPhysics(),
           itemCount: category.quizquestionsets.length,

          itemBuilder:(context,index){
           

            final quizset= category.quizquestionsets[index];
            return Padding(padding: 
            EdgeInsets.all(10.0),
          child:   GestureDetector(onTap:(){
               Navigator.push(context ,MaterialPageRoute(builder:(context)=>Quizpage(quizset:quizset)) );



          },
          child:Container(
            width:MediaQuery.of(context).size.width/1
            ,
            padding: EdgeInsets.symmetric(vertical: 10,horizontal: 15),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.2),
              borderRadius: BorderRadius.circular(25),
            ),
            child: Row(
              children: [
               Icon(Icons.ac_unit_rounded,size:30),
                SizedBox(width:20),
                Text( quizset.name,
                style: TextStyle(
                  fontSize: 18,
                  color:Colors.white,
 fontWeight: FontWeight.w500  ,
              ),

                )

              ]
            ),
          ) ,)
            );

          }

          

         )
          
            ],
          ),
        ),
      ),
    );
  }
}
