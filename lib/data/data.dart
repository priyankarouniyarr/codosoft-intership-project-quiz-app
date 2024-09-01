import 'package:quiz_app/models/models.dart';

class Data {
  final List<Category> categories = [
    Category(
      name: 'Artifical Intelligence',
      image: 'images/artifical_intelligence.png',
      quizquestionsets: [
        QuizSet(
          name: 'Beginner',
          questions: [
            Question(
                '1. What is Artificial Intelligence?',
               [
                  'A) Programming',
                  'B) Machine learning',
                  'C) Simulation of human intelligence',
                  'D) Data processing'
                ],
                2),
            Question(
                '2. Which of the following is an application of AI?',
                [
                  'A) Speech recognition',
                  'B) Email',
                  'C) Social media',
                  'D) Web browsing'
                ],
                0),
            Question(
                '3. Who is known as the father of AI?',
                [
                  'A) Alan Turing',
                  'B) John McCarthy',
                  'C) Geoffrey Hinton',
                  'D) Elon Musk'
                ],
                1),
            Question(
                '4. What does NLP stand for in AI?',
                [
                  'A) Natural Language Processing',
                  'B) Neural Link Protocol',
                  'C) Network Language Processing',
                  'D) Natural Learning Process'
                ],
                0),
            Question(
                '5. Which of the following is a type of AI?',
                [
                  'A) Weak AI',
                  'B) Strong AI',
                  'C) Super AI',
                  'D) All of the above'
                ],
                3),
            Question(
                '6. What is a chatbot?',
                [
                  'A) A program that can have conversations',
                  'B) A type of robot',
                  'C) A machine learning model',
                  'D) A web browser extension'
                ],
                0),
            Question(
                '7. Which of the following is a common AI programming language?',
                ['A) Python', 'B) HTML', 'C) CSS', 'D) SQL'],
                0),
            Question(
                '8. What is machine learning?',
                [
                  'A) A subset of AI',
                  'B) A programming language',
                  'C) A hardware component',
                  'D) A type of computer network'
                ],
                0),
            Question(
                '9. What does GPU stand for in AI processing?',
                [
                  'A) General Processing Unit',
                  'B) Graphical Processing Unit',
                  'C) Graphics Performance Utility',
                  'D) General Purpose Unit'
                ],
                2),
            Question(
                '10. What is deep learning?',
                [
                  'A) A type of neural network',
                  'B) A programming language',
                  'C) An AI technique that mimics the human brain',
                  'D) Both A and C'
                ],
                3),
            Question('11. Which of the following is a popular AI framework?',
                ['A) TensorFlow', 'B) Bootstrap', 'C) React', 'D) Laravel'], 0),
            Question(
                '12. What is reinforcement learning?',
                [
                  'A) A type of supervised learning',
                  'B) A type of unsupervised learning',
                  'C) Learning based on rewards and punishments',
                  'D) A type of deep learning'
                ],
                2),
            Question(
                '13. Which of the following is an example of supervised learning?',
                [
                  'A) Clustering',
                  'B) Classification',
                  'C) Dimensionality reduction',
                  'D) Data compression'
                ],
                1),
            Question(
                '14. What is the Turing Test?',
                [
                  'A) A test to measure machine intelligence',
                  'B) A test for software bugs',
                  'C) A computer performance test',
                  'D) A test for internet speed'
                ],
                0),
            Question('15. Which of the following is used to train AI models?',
                ['A) Data', 'B) RAM', 'C) CPU', 'D) Internet'], 0),
            Question(
                '16. What does the term “overfitting” refer to in AI?',
                [
                  'A) A model that is too simple',
                  'B) A model that performs well on training data but poorly on new data',
                  'C) A model with too many layers',
                  'D) A model with too much data'
                ],
                1),
            Question(
                '17. What is the primary goal of AI?',
                [
                  'A) To replace humans',
                  'B) To create intelligent systems that can perform tasks autonomously',
                  'C) To entertain people',
                  'D) To create video games'
                ],
                1),
            Question(
                '18. Which AI concept involves the computer understanding and responding to human speech?',
                [
                  'A) Speech synthesis',
                  'B) Voice modulation',
                  'C) Speech recognition',
                  'D) Voice over'
                ],
                2),
            Question(
                '19. What is an artificial neural network?',
                [
                  'A) A software framework',
                  'B) A collection of algorithms',
                  'C) A system modeled after the human brain',
                  'D) A hardware component'
                ],
                2),
            Question(
                '20. Which of the following best describes “Big Data” in AI?',
                [
                  'A) Large datasets used for training AI models',
                  'B) A software tool',
                  'C) A programming language',
                  'D) A type of computer memory'
                ],
                0),
          ],
        ),
        QuizSet(
          name: 'Intermediate',
          questions: [
            Question(
                '1. What does AI stand for?',
                [
                  'A) Automated Intelligence',
                  'B) Artificial Intelligence',
                  'C) Advanced Integration',
                  'D) Applied Informatics'
                ],
                1),
            Question(
                '2. What is the primary goal of a neural network in AI?',
                [
                  'A) Data storage',
                  'B) Pattern recognition',
                  'C) Network management',
                  'D) System security'
                ],
                1),
            Question(
                '3. Which algorithm is commonly used for classification tasks in machine learning?',
                [
                  'A) Linear regression',
                  'B) Decision trees',
                  'C) k-means clustering',
                  'D) Principal Component Analysis'
                ],
                1),
            Question(
                '4. What does the term "overfitting" refer to in machine learning?',
                [
                  'A)   Excessive complexity of the model',
                  'B)Model accuracy',
                  'C) Insufficient training data',
                  'D) Model simplicity'
                ],
                0),
            Question(
                '5. In the context of neural networks, what does "backpropagation" do?',
                [
                  'A) Forward pass of data',
                  'B) Optimizes weights and biases',
                  'C) Initializes network parameters',
                  'D) Validates model performance'
                ],
                1),
            Question(
                '6. Which of the following is a type of unsupervised learning?',
                [
                  'A) Linear regression',
                  'B) Decision trees',
                  'C) k-means clustering',
                  'D) Naive Bayes'
                ],
                2),
            Question(
                '7. What is the main purpose of the "loss function" in machine learning?',
                [
                  'A) To evaluate model accuracy',
                  'B) To optimize the learning rate',
                  'C) To measure prediction error',
                  'D) To initialize model parameters'
                ],
                2),
            Question(
                '8. What does "feature engineering" involve in machine learning?',
                [
                  'A) Data cleaning',
                  'B) Model evaluation',
                  'C) Creating new features from existing data',
                  'D) Training data selection'
                ],
                2),
            Question(
                '9. What is "ensemble learning" in the context of AI?',
                [
                  'A) Using multiple models to improve performance',
                  'B) A single model approach',
                  'C) Data preprocessing',
                  'D) Hyperparameter tuning'
                ],
                0),
            Question(
                '10. Which of the following is NOT a type of neural network architecture?',
                [
                  'A) Convolutional Neural Network (CNN)',
                  'B) Recurrent Neural Network (RNN)',
                  'C) Support Vector Machine (SVM)',
                  'D) Generative Adversarial Network (GAN)'
                ],
                2),
            Question(
                '11. In AI, what does "NLP" stand for?',
                [
                  'A) Natural Language Processing',
                  'B) Neural Language Protocol',
                  'C) Network Language Processing',
                  'D) Numerical Language Programming'
                ],
                0),
            Question(
                '12. What is the primary purpose of "data normalization" in machine learning?',
                [
                  'A) To improve model performance',
                  'B) To increase data volume',
                  'C) To handle missing values',
                  'D) To reduce data dimensionality'
                ],
                0),
            Question(
                '13. Which type of machine learning is used when the output is known and the model learns from labeled data?',
                [
                  'A) Unsupervised Learning',
                  'B) Reinforcement Learning',
                  'C) Supervised Learning',
                  'D) Semi-supervised Learning'
                ],
                2),
            Question(
                '14. What is "gradient descent" used for in machine learning?',
                [
                  'A) To optimize the model parameters',
                  'B) To initialize weights',
                  'C) To evaluate model accuracy',
                  'D) To preprocess data'
                ],
                0),
            Question(
                '15. Which of the following methods is used to prevent overfitting in a model?',
                [
                  'A) Cross-validation',
                  'B) Feature selection',
                  'C) Data augmentation',
                  'D) All of the above'
                ],
                3),
            Question(
                '16. What is the role of "activation functions" in a neural network?',
                [
                  'A) To introduce non-linearity',
                  'B) To initialize weights',
                  'C) To perform data normalization',
                  'D) To evaluate model performance'
                ],
                0),
          ],
        ),
        QuizSet(
          name: 'Difficult',
          questions: [
            Question(
                '1. What is the primary function of the "Softmax" activation function?',
                [
                  'A) To normalize outputs to a probability distribution',
                  'B) To reduce dimensionality',
                  'C) To handle missing values',
                  'D) To initialize network weights'
                ],
                0),
            Question(
                '2. Which of the following is an example of a generative model?',
                [
                  'A) Generative Adversarial Networks (GANs)',
                  'B) Support Vector Machines',
                  'C) Decision Trees',
                  'D) Random Forests'
                ],
                0),
            Question(
                '3. In the context of deep learning, what is "dropout" used for?',
                [
                  'A) To prevent overfitting',
                  'B) To increase the model capacity',
                  'C) To speed up training',
                  'D) To handle large datasets'
                ],
                0),
            Question(
                '4. Which algorithm is primarily used for dimensionality reduction?',
                [
                  'A) Principal Component Analysis (PCA)',
                  'B) k-Nearest Neighbors',
                  'C) Linear Regression',
                  'D) Naive Bayes'
                ],
                0),
            Question(
                '5. What is "reinforcement learning" most commonly used for?',
                [
                  'A) Sequential decision-making tasks',
                  'B) Classification tasks',
                  'C) Supervised learning tasks',
                  'D) Data preprocessing'
                ],
                0),
            Question(
                '6. What is "regularization" in machine learning used for?',
                [
                  'A) To prevent overfitting',
                  'B) To enhance model interpretability',
                  'C) To improve computational efficiency',
                  'D) To balance the dataset'
                ],
                0),
            Question(
                '7. Which of the following techniques is used to handle class imbalance in a dataset?',
                [
                  'A)Hyperparameter tuning ',
                  'B) Cross-validation',
                  'C) Feature scaling',
                  'D)  Data augmentation'
                ],
                3),
            Question(
                '8. In which type of neural network is "convolution" typically used?',
                [
                  'A) Convolutional Neural Networks (CNNs)',
                  'B) Recurrent Neural Networks (RNNs)',
                  'C) Long Short-Term Memory (LSTM) Networks',
                  'D) Generative Adversarial Networks (GANs)'
                ],
                0),
            Question(
                '9. What does the "bias" parameter in a neural network do?',
                [
                  'A) Adjusts the output independently of the input',
                  'B) Normalizes the input data',
                  'C) Introduces non-linearity',
                  'D) Optimizes the learning rate'
                ],
                0),
            Question(
                '10. Which metric is commonly used to evaluate the performance of a classification model?',
                [
                  'A) Mean Squared Error (MSE)',
                  'B) Mean Absolute Error (MAE)',
                  'C)  F1 Score',
                  'D) R-Squared'
                ],
                2 // answer
                ),
          ],
        ),
      ],
    ),
    Category(
      name: 'Data Science and Analytics',
      image: 'images/data science.jpg',
      quizquestionsets: [
        QuizSet(
          name: 'Beginner',
          questions: [
            Question(
                '1. What is the primary goal of Data Science?',
                [
                  'A) To extract insights and knowledge from data',
                  'B) To create web applications',
                  'C) To design user interfaces',
                  'D) To develop hardware components'
                ],
                2),
            Question(
                '2. What does "EDA" stand for in Data Science?',
                [
                  'A) Exploratory Data Analysis',
                  'B) Effective Data Assessment',
                  'C) Enhanced Data Application',
                  'D) Empirical Data Analysis'
                ],
                1),
            Question(
                '3. Which of the following is a commonly used programming language in Data Science?',
                ['A) Python', 'B) JavaScript', 'C) HTML', 'D) CSS'],
                3),
            Question(
                '4. What is "data cleaning" in the context of data analysis?',
                [
                  'A) Removing or correcting errors in data',
                  'B) Organizing data into tables',
                  'C) Analyzing trends in data',
                  'D) Visualizing data patterns'
                ],
                1),
            Question(
                '5. Which library is commonly used for data manipulation and analysis in Python?',
                ['A) Pandas', 'B) TensorFlow', 'C) Flask', 'D) NumPy'],
                2),
            Question(
                '6. What does "SQL" stand for?',
                [
                  'A) Structured Query Language',
                  'B) Simple Query Language',
                  'C) Standard Query Language',
                  'D) Statistical Query Language'
                ],
                3),
            Question(
                '7. What is a "dataset" in Data Science?',
                [
                  'A) A collection of related data',
                  'B) A single data point',
                  'C) A data visualization',
                  'D) A type of database'
                ],
                0),
            Question(
                '8. What is "regression analysis" used for?',
                [
                  'A) To model and analyze relationships between variables',
                  'B) To classify data into categories',
                  'C) To reduce data dimensionality',
                  'D) To cluster similar data points'
                ],
                2),
            Question(
                '9. Which visualization tool is popular for creating interactive dashboards?',
                ['A) Tableau', 'B) Visual Studio', 'C) PyCharm', 'D) Eclipse'],
                1),
            Question(
                '10. What does "machine learning" involve?',
                [
                  'A) Training algorithms to learn from data and make predictions',
                  'B) Writing code to automate tasks',
                  'C) Designing user interfaces',
                  'D) Managing databases'
                ],
                3),
          ],
        ),
        QuizSet(
          name: 'Intermediate',
          questions: [
            Question(
                '1. What is "feature engineering" in the context of machine learning?',
                [
                  'A) Creating new features from raw data',
                  'B) Choosing machine learning algorithms',
                  'C) Tuning model hyperparameters',
                  'D) Evaluating model performance'
                ],
                0),
            Question(
                '2. What is the purpose of a confusion matrix in classification tasks?',
                [
                  'A) To evaluate the performance of a classification model',
                  'B) To preprocess data for training',
                  'C) To select features for the model',
                  'D) To visualize the data distribution'
                ],
                0),
            Question(
                '3. In which scenario would you use a "random forest" algorithm?',
                [
                  'A) When you need to perform classification or regression with high accuracy',
                  'B) When you need to cluster data points',
                  'C) When you want to perform dimensionality reduction',
                  'D) When you want to calculate correlations'
                ],
                1),
            Question(
                '4. What does "overfitting" refer to in machine learning?',
                [
                  'A) When a model performs well on training data but poorly on test data',
                  'B) When a model performs equally well on both training and test data',
                  'C) When a model has too few parameters',
                  'D) When a model is too simple'
                ],
                1),
            Question(
                '5. What is the "bias-variance tradeoff" in machine learning?',
                [
                  'A) The balance between model complexity and error',
                  'B) The tradeoff between training and testing accuracy',
                  'C) The selection of features versus model training',
                  'D) The choice between supervised and unsupervised learning'
                ],
                0),
            Question(
                '6. What is "cross-validation" used for?',
                [
                  'A) To assess the performance of a model on unseen data',
                  'B) To increase the size of the training dataset',
                  'C) To tune the hyperparameters of a model',
                  'D) To preprocess data for analysis'
                ],
                2),
            Question(
                '7. What is the "K-nearest neighbors" (KNN) algorithm used for?',
                [
                  'A) Classification and regression tasks based on similarity',
                  'B) Data preprocessing and cleaning',
                  'C) Feature selection and extraction',
                  'D) Dimensionality reduction'
                ],
                1),
            Question(
                '8. Which method is commonly used for dimensionality reduction?',
                [
                  'A) Principal Component Analysis (PCA)',
                  'B) Linear Regression',
                  'C) Decision Trees',
                  'D) Neural Networks'
                ],
                1),
          ],
        ),
        QuizSet(
          name: 'Difficult',
          questions: [
            Question(
                '1. What is "Bayesian inference" used for in data science?',
                [
                  'A) To update the probability of a hypothesis based on new evidence',
                  'B) To reduce the dimensionality of data',
                  'C) To evaluate the model performance',
                  'D) To perform feature scaling'
                ],
                2),
            Question(
                '2. What does "Gradient Boosting" involve in machine learning?',
                [
                  'A) Sequentially correcting errors of a series of weak models',
                  'B) Training multiple models independently and averaging their outputs',
                  'C) Using neural networks for feature extraction',
                  'D) Performing dimensionality reduction'
                ],
                1),
            Question(
                '3. What is the "curse of dimensionality" in data analysis?',
                [
                  'A) The challenge that data becomes sparse as dimensions increase',
                  'B) The difficulty of having too few features in a dataset',
                  'C) The problem of overfitting in low-dimensional spaces',
                  'D) The issue of data loss during transformation'
                ],
                1),
            Question(
                '4. In time series analysis, what does "seasonality" refer to?',
                [
                  'A) Repeating patterns or cycles in data over specific time intervals',
                  'B) Random noise in the data',
                  'C) Long-term trends in the data',
                  'D) Irregular variations not predictable'
                ],
                0),
            Question(
                '5. What is "Regularization" in the context of machine learning models?',
                [
                  'A) Techniques to prevent overfitting by penalizing large coefficients',
                  'B) Methods to optimize model hyperparameters',
                  'C) Procedures to normalize data distributions',
                  'D) Approaches to increase model training speed'
                ],
                2),
            Question(
                '6. What is the purpose of "t-SNE" in data analysis?',
                [
                  'A) To visualize high-dimensional data in a lower-dimensional space',
                  'B) To perform feature selection',
                  'C) To build predictive models',
                  'D) To handle missing values'
                ],
                1),
            Question(
                '7. What is "Expectation-Maximization (EM)" algorithm used for?',
                [
                  'A) To find maximum likelihood estimates of parameters in models with hidden variables',
                  'B) To train neural networks',
                  'C) To perform cross-validation',
                  'D) To perform dimensionality reduction'
                ],
                0),
            Question(
                '8. What does "LSTM" stand for in neural networks?',
                [
                  'A) Long Short-Term Memory',
                  'B) Linear Support Transform Model',
                  'C) Low Short-Term Memory',
                  'D) Large Scale Temporal Model'
                ],
                2),
            Question(
                '9. What is the "AUC-ROC" curve used to evaluate?',
                [
                  'A) The performance of a classification model by plotting true positive rate versus false positive rate',
                  'B) The accuracy of a regression model',
                  'C) The efficiency of feature selection',
                  'D) The stability of clustering algorithms'
                ],
                3),
            Question(
                '10. What is "Hyperparameter Tuning" in machine learning?',
                [
                  'A) The process of selecting the best hyperparameters for a learning algorithm',
                  'B) The technique of adjusting data distributions',
                  'C) The method for evaluating model performance',
                  'D) The way to preprocess input features'
                ],
                1),
          ],
        ),
      ],
    ),
    Category(
      name: 'General Knowledge',
      image: 'images/gk.png',
      quizquestionsets: [
        QuizSet(
          name: 'Beginner',
          questions: [
            Question('1. What is the capital of France?',
                ['A) Berlin', 'B) Madrid', 'C) Paris', 'D) Rome'], 2),
            Question(
                '2. Who wrote the play "Romeo and Juliet"?',
                [
                  'A) William Shakespeare',
                  'B) Charles Dickens',
                  'C) Jane Austen',
                  'D) Mark Twain'
                ],
                0),
            Question('3. What is the largest planet in our solar system?',
                ['A) Earth', 'B) Mars', 'C) Jupiter', 'D) Saturn'], 2),
            Question('4. What is the chemical symbol for water?',
                ['A) O2', 'B) CO2', 'C) H2O', 'D) NaCl'], 2),
            Question(
                '5. Who painted the Mona Lisa?',
                [
                  'A) Vincent van Gogh',
                  'B) Leonardo da Vinci',
                  'C) Pablo Picasso',
                  'D) Claude Monet'
                ],
                1),
            Question('6. What is the hardest natural substance on Earth?',
                ['A) Gold', 'B) Iron', 'C) Diamond', 'D) Platinum'], 2),
            Question('7. How many continents are there on Earth?',
                ['A) 5', 'B) 6', 'C) 7', 'D) 8'], 2),
            Question(
                '8. Who was the first President of the United States?',
                [
                  'A) Abraham Lincoln',
                  'B) George Washington',
                  'C) Thomas Jefferson',
                  'D) John Adams'
                ],
                1),
            Question('9. What is the main ingredient in guacamole?',
                ['A) Tomato', 'B) Onion', 'C) Avocado', 'D) Pepper'], 2),
            Question(
                '10. What is the largest ocean on Earth?',
                [
                  'A) Atlantic Ocean',
                  'B) Indian Ocean',
                  'C) Arctic Ocean',
                  'D) Pacific Ocean'
                ],
                3),
            Question(
                '11. What gas do plants use for photosynthesis?',
                [
                  'A) Oxygen',
                  'B) Nitrogen',
                  'C) Carbon Dioxide',
                  'D) Hydrogen'
                ],
                2),
            Question(
                '12. Who is known as the "Father of Computers"?',
                [
                  'A) Charles Babbage',
                  'B) Alan Turing',
                  'C) Bill Gates',
                  'D) Steve Jobs'
                ],
                0),
            Question('13. What is the currency used in Japan?',
                ['A) Yen', 'B) Won', 'C) Dollar', 'D) Euro'], 0),
            Question(
                '14. What is the largest mammal in the world?',
                [
                  'A) Elephant',
                  'B) Blue Whale',
                  'C) Giraffe',
                  'D) Hippopotamus'
                ],
                1),
            Question('15. What is the smallest prime number?',
                ['A) 0', 'B) 1', 'C) 2', 'D) 3'], 2),
          ],
        ),
        QuizSet(
          name: 'Intermediate',
          questions: [
            Question('1. What is the chemical symbol for gold?',
                ['A) Au', 'B) Ag', 'C) Pb', 'D) Fe'], 0),
            Question(
                '2. Who wrote "The Odyssey"?',
                ['A) Homer', 'B) Virgil', 'C) Sophocles', 'D) Aristophanes'],
                0),
            Question(
                '3. What is the longest river in the world?',
                [
                  'A) Amazon River',
                  'B) Nile River',
                  'C) Yangtze River',
                  'D) Mississippi River'
                ],
                1),
            Question(
                '4. Who is known as the "Father of Modern Physics"?',
                [
                  'A) Isaac Newton',
                  'B) Albert Einstein',
                  'C) Niels Bohr',
                  'D) Galileo Galilei'
                ],
                1),
            Question('5. What is the capital city of Canada?',
                ['A) Toronto', 'B) Vancouver', 'C) Montreal', 'D) Ottawa'], 3),
            Question(
                '6. What is the largest desert in the world?',
                [
                  'A) Sahara Desert',
                  'B) Arabian Desert',
                  'C) Gobi Desert',
                  'D) Antarctic Desert'
                ],
                3),
            Question(
                '7. Who was the first woman to win a Nobel Prize?',
                [
                  'A) Marie Curie',
                  'B) Rosalind Franklin',
                  'C) Ada Lovelace',
                  'D) Barbara McClintock'
                ],
                0),
            Question('8. In which year did the Titanic sink?',
                ['A) 1905', 'B) 1912', 'C) 1918', 'D) 1920'], 1),
            Question('9. What is the currency of Brazil?',
                ['A) Peso', 'B) Real', 'C) Dollar', 'D) Euro'], 1),
            Question(
                '10. Who painted "The Persistence of Memory"?',
                [
                  'A) Salvador Dalí',
                  'B) Pablo Picasso',
                  'C) Vincent van Gogh',
                  'D) Claude Monet'
                ],
                0),
          ],
        ),
        QuizSet(
          name: 'Difficult',
          questions: [
            Question(
                '1. Who was the first Emperor of China?',
                [
                  'A) Qin Shi Huang',
                  'B) Han Wudi',
                  'C) Tang Taizong',
                  'D) Kangxi'
                ],
                0),
            Question('2. What is the only continent without a desert?',
                ['A) Europe', 'B) Africa', 'C) Asia', 'D) South America'], 0),
            Question(
                '3. Which scientist developed the theory of general relativity?',
                [
                  'A) Isaac Newton',
                  'B) Albert Einstein',
                  'C) Niels Bohr',
                  'D) James Clerk Maxwell'
                ],
                1),
            Question('4. What is the smallest bone in the human body?',
                ['A) Stapes', 'B) Incus', 'C) Malleus', 'D) Femur'], 0),
            Question('5. In what year was the United Nations established?',
                ['A) 1942', 'B) 1945', 'C) 1950', 'D) 1955'], 1),
            Question(
                '6. Who was the first person to reach the South Pole?',
                [
                  'A) Robert Falcon Scott',
                  'B) Ernest Shackleton',
                  'C) Roald Amundsen',
                  'D) Edmund Hillary'
                ],
                2),
            Question('7. What is the capital city of Bhutan?',
                ['A) Kathmandu', 'B) Thimphu', 'C) Dhaka', 'D) Colombo'], 1),
            Question(
                '8. Who wrote the novel "One Hundred Years of Solitude"?',
                [
                  'A) Gabriel García Márquez',
                  'B) Mario Vargas Llosa',
                  'C) Julio Cortázar',
                  'D) Carlos Fuentes'
                ],
                0),
          ],
        ),
      ],
    ),
    Category(
      name: 'Geography',
      image: 'images/geography.png',
      quizquestionsets: [
        QuizSet(
          name: 'Beginner',
          questions: [
            Question('1. What is the largest continent by land area?',
                ['A) Africa', 'B) Asia', 'C) Europe', 'D) North America'], 1),
            Question('2. Which river is the longest in the world?',
                ['A) Nile', 'B) Amazon', 'C) Yangtze', 'D) Mississippi'], 1),
            Question('3. What is the capital city of France?',
                ['A) Paris', 'B) Rome', 'C) Madrid', 'D) Berlin'], 0),
            Question(
                '4. Which ocean is the largest?',
                [
                  'A) Atlantic Ocean',
                  'B) Indian Ocean',
                  'C) Arctic Ocean',
                  'D) Pacific Ocean'
                ],
                3),
            Question(
                '5. What is the smallest country in the world by land area?',
                [
                  'A) Monaco',
                  'B) Vatican City',
                  'C) San Marino',
                  'D) Liechtenstein'
                ],
                1),
            Question('6. Which country is known as the Land of the Rising Sun?',
                ['A) China', 'B) South Korea', 'C) Japan', 'D) Thailand'], 2),
            Question(
                '7. What is the largest desert in the world?',
                [
                  'A) Sahara Desert',
                  'B) Arabian Desert',
                  'C) Gobi Desert',
                  'D) Kalahari Desert'
                ],
                0),
            Question('8. Which mountain range is the highest in the world?',
                ['A) Andes', 'B) Alps', 'C) Himalayas', 'D) Rockies'], 2),
            Question('9. What is the official language of Brazil?',
                ['A) Spanish', 'B) Portuguese', 'C) English', 'D) French'], 1),
            Question(
                '10. Which continent is known for having the most number of countries?',
                ['A) Africa', 'B) Asia', 'C) Europe', 'D) South America'],
                0),
          ],
        ),
        QuizSet(
          name: 'Intermediate',
          questions: [
            Question(
                '1. Which country is known as the Land of the Midnight Sun?',
                ['A) Norway', 'B) Sweden', 'C) Finland', 'D) Russia'],
                0),
            Question('2. What is the capital city of Australia?',
                ['A) Sydney', 'B) Melbourne', 'C) Canberra', 'D) Brisbane'], 2),
            Question(
                '3. Which desert is located in northern China and southern Mongolia?',
                ['A) Kalahari', 'B) Atacama', 'C) Gobi', 'D) Namib'],
                2),
            Question(
                '4. Which river flows through Egypt and is the primary source of water for the country?',
                ['A) Amazon', 'B) Nile', 'C) Yangtze', 'D) Tigris'],
                1),
            Question('5. What is the most populous city in the world?',
                ['A) Tokyo', 'B) Shanghai', 'C) Delhi', 'D) São Paulo'], 0),
            Question(
                '6. Which mountain range forms the natural border between Europe and Asia?',
                ['A) Rockies', 'B) Alps', 'C) Ural Mountains', 'D) Andes'],
                2),
            Question(
                '7. What is the name of the longest river in South America?',
                ['A) Orinoco', 'B) Paraná', 'C) Amazon', 'D) São Francisco'],
                2),
            Question(
                '8. Which city is known as the City of Canals?',
                ['A) Venice', 'B) Amsterdam', 'C) Bangkok', 'D) Copenhagen'],
                0),
            Question(
                '9. Which African country has the highest population density?',
                ['A) Nigeria', 'B) Egypt', 'C) Rwanda', 'D) Kenya'],
                2),
            Question(
                '10. Which sea is the world\'s largest inland body of water?',
                [
                  'A) Caspian Sea',
                  'B) Lake Superior',
                  'C) Dead Sea',
                  'D) Lake Victoria'
                ],
                0),
          ],
        ),
        QuizSet(
          name: 'Difficult',
          questions: [
            Question(
                '1. What is the deepest point in the world’s oceans?',
                [
                  'A)Puerto Rico Trench',
                  'B) Tonga Trench',
                  'C)   Mariana Trench',
                  'D) Java Trench'
                ],
                2),
            Question(
                '2. Which is the only country to have a coastline on both the Atlantic and Indian Oceans?',
                [
                  'A)Mozambique ',
                  'B)  South Africa',
                  'C) Tanzania',
                  'D) Kenya'
                ],
                1),
            Question(
                '3. What is the capital of the only landlocked country in South America?',
                ['A) La Paz', 'B) Quito', 'C) Bogotá', 'D) Asunción'],
                0),
            Question(
                '4. Which country has the longest coastline in the world?',
                ['A) United States', 'B) Russia', 'C) Australia', 'D)  Canada'],
                3),
            Question(
                '5. What is the smallest country by land area in the world?',
                [
                  'A) Monaco',
                  'B) San Marino',
                  'C) Vatican City',
                  'D) Liechtenstein'
                ],
                2),
          ],
        ),
      ],
    ),
    Category(
      name: 'Software Development',
      image: 'images/software.jpg',
      quizquestionsets: [
        QuizSet(
          name: 'Beginner',
          questions: [
            Question(
                '1. What does IDE stand for in software development?',
                [
                  'A) Integrated Development Environment',
                  'B) International Development Environment',
                  'C) Interactive Design Editor',
                  'D) Internal Development Engine'
                ],
                0),
            Question(
                '2. Which programming language is commonly used for web development?',
                ['A) Python', 'B) JavaScript', 'C) Java', 'D) C++'],
                1),
            Question(
                '3. What is a "bug" in software development?',
                [
                  'A) A feature',
                  'B) A design pattern',
                  'C) An error or defect',
                  'D) A programming language'
                ],
                3),
            Question(
                '4. Which version control system is widely used for managing source code?',
                ['A) Git', 'B) SVN', 'C) CVS', 'D) FTP'],
                0),
            Question(
                '5. What is the purpose of a "compiler" in programming?',
                [
                  'A) To write code',
                  'B) To debug code',
                  'C) To translate code into machine language',
                  'D) To manage databases'
                ],
                2),
            Question(
                '6. What does "API" stand for in software development?',
                [
                  'A) Application Programming Interface',
                  'B) Advanced Programming Integration',
                  'C) Automated Process Interface',
                  'D) Application Product Information'
                ],
                0),
            Question(
                '7. What is the purpose of "unit testing" in software development?',
                [
                  'A) To test the entire system',
                  'B) To test individual units or components of code',
                  'C) To test user interfaces',
                  'D) To test deployment processes'
                ],
                1),
            Question(
                '8. What does "debugging" refer to in programming?',
                [
                  'A) Writing new code',
                  'B) Removing errors from code',
                  'C) Compiling code',
                  'D) Designing user interfaces'
                ],
                1),
            Question(
                '9. What is "object-oriented programming"?',
                [
                  'A) A programming paradigm based on objects and classes',
                  'B) A type of database management',
                  'C) A tool for code deployment',
                  'D) A method for writing algorithms'
                ],
                0),
            Question(
                '10. What is the primary purpose of a "database" in software applications?',
                [
                  'A) To execute code',
                  'B) To store and manage data',
                  'C) To create user interfaces',
                  'D) To compile programs'
                ],
                1),
            Question(
                '11. What does "UI" stand for in the context of software development?',
                [
                  'A) User Interface',
                  'B) Universal Integration',
                  'C) Unified Information',
                  'D) User Interaction'
                ],
                0),
            Question(
                '12. What is "source code"?',
                [
                  'A) The compiled output of a program',
                  'B) The original code written by a programmer',
                  'C) A type of database file',
                  'D) A software license agreement'
                ],
                1),
          ],
        ),
        QuizSet(
          name: 'Intermediate',
          questions: [
            Question(
                '1. What is the main purpose of using a "design pattern" in software development?',
                [
                  'A) To improve code readability',
                  'B) To provide solutions to common design problems',
                  'C) To increase code execution speed',
                  'D) To manage database connections'
                ],
                1),
            Question(
                '2. What is "continuous integration" (CI) in the context of software development?',
                [
                  'A) Integrating new features into a codebase once a week',
                  'B) Regularly merging code changes into a shared repository',
                  'C) Conducting security audits on code',
                  'D) Testing software on multiple platforms'
                ],
                1),
            Question(
                '3. What does "refactoring" involve in software development?',
                [
                  'A) Adding new features to an application',
                  'B) Improving the structure of existing code without changing its functionality',
                  'C) Writing new test cases',
                  'D) Upgrading the hardware on which the application runs'
                ],
                1),
            Question(
                '4. In "Agile" methodology, what is the primary purpose of a "sprint"?',
                [
                  'A) To deliver a complete product',
                  'B)  To review project milestones ',
                  'C) To test software performance',
                  'D)To complete a specific set of features within a set time frame'
                ],
                3),
            Question(
                '5. What is a "middleware" in software architecture?',
                [
                  'A) A layer of software that connects different applications or services',
                  'B) A tool for debugging code',
                  'C) A programming language',
                  'D) A type of database'
                ],
                0),
            Question(
                '6. What is "Dependency Injection" (DI) used for in software development?',
                [
                  'A) To automatically update dependencies in code',
                  'B) To reduce the dependency of components on each other',
                  'C) To increase the speed of data processing',
                  'D) To manage database transactions'
                ],
                1),
            Question(
                '7. What is the purpose of using "mock objects" in unit testing?',
                [
                  'A) To simulate real objects for testing purposes',
                  'B) To speed up the testing process',
                  'C) To provide real data for testing',
                  'D) To replace a testing framework'
                ],
                0),
            Question(
                '8. What does "CI/CD" stand for in the context of modern software development practices?',
                [
                  'A) Continuous Integration/Continuous Deployment',
                  'B) Continuous Improvement/Continuous Delivery',
                  'C) Code Integration/Code Deployment',
                  'D) Continuous Inspection/Continuous Debugging'
                ],
                0),
            Question(
                '9. What is "containerization" in software development?',
                [
                  'A) Packaging software to run in isolated environments',
                  'B) Storing data in containers',
                  'C) Managing cloud resources',
                  'D) Debugging container-related issues'
                ],
                0),
          ],
        ),
        QuizSet(
          name: 'Difficult',
          questions: [
            Question(
                '1. What is "Lazy Evaluation" in programming?',
                [
                  'A) Delaying the evaluation of an expression until its value is actually needed',
                  'B) Evaluating all expressions at once for better performance',
                  'C) Ignoring errors during code execution',
                  'D) Automatically optimizing code execution'
                ],
                0),
            Question(
                '2. In the context of database normalization, what is the main purpose of the Third Normal Form (3NF)?',
                [
                  'A) To eliminate transitive dependencies',
                  'B) To ensure data is stored in a single table',
                  'C) To ensure that all columns are dependent on the primary key',
                  'D) To separate data into multiple tables'
                ],
                0),
            Question(
                '3. What does "CAP Theorem" stand for in distributed systems?',
                [
                  'A) Consistency, Access, Performance  ',
                  'B) Complexity, Accuracy, Performance',
                  'C)  Consistency, Availability, Partition Tolerance',
                  'D) Capability, Access, Partition'
                ],
                2),
            Question(
                '4. In the context of machine learning, what is "Bias-Variance Tradeoff"?',
                [
                  'A) The balance between a model’s ability to generalize and its sensitivity to training data',
                  'B) The tradeoff between model accuracy and data processing speed',
                  'C) The tradeoff between bias in training data and model performance',
                  'D) The balance between the number of features and the training dataset size'
                ],
                0),
            Question(
                '5. What is the purpose of "Backpropagation" in neural networks?',
                [
                  'A)To initialize network weights ',
                  'B)  To update the weights of neurons based on the error',
                  'C) To create new neurons',
                  'D) To manage network connections'
                ],
                1),
            Question(
                '6. What is a "Semaphore" used for in concurrent programming?',
                [
                  'A) To handle error ',
                  'B) To handle exceptions during program execution',
                  'C) To control access to a common resource by multiple processes'
                ],
                3),
            Question(
                '7. In software development, what is the "Observer Pattern"?',
                [
                  'A) A design pattern where an object maintains a list of its dependents and notifies them of state changes',
                  'B) A pattern for managing database connections',
                  'C) A method for sorting and searching data',
                  'D) A strategy for handling user input'
                ],
                0),
          ],
        ),
      ],
    ),
    Category(
      name: 'Business and Economy',
      image: 'images/business.png',
      quizquestionsets: [
        QuizSet(
          name: 'Beginner',
          questions: [
            Question(
                '1. What is the primary goal of a business?',
                [
                  'A) To make a profit',
                  'B) To increase market share',
                  'C) To expand internationally',
                  'D) To reduce costs'
                ],
                1),
            Question(
                '2. What is GDP an acronym for?',
                [
                  'A) Gross Domestic Product',
                  'B) General Domestic Product',
                  'C) Gross Development Program',
                  'D) General Development Program'
                ],
                3),
            Question(
                '3. What does ROI stand for in business?',
                [
                  'A) Return on Investment',
                  'B) Rate of Interest',
                  'C) Return on Income',
                  'D) Rate of Investment'
                ],
                2),
            Question(
                '4. Which term describes the total market value of all goods and services produced in a country?',
                ['A) GDP', 'B) GNP', 'C) CPI', 'D) PPI'],
                0),
            Question(
                '5. What is the purpose of a business plan?',
                [
                  'A) To outline a company’s goals and strategies',
                  'B) To track daily expenses',
                  'C) To manage employee schedules',
                  'D) To handle customer complaints'
                ],
                3),
            Question(
                '6. Which financial statement shows a company’s profitability over a period of time?',
                [
                  'A) Balance Sheet',
                  'B) Income Statement',
                  'C) Cash Flow Statement',
                  'D) Statement of Retained Earnings'
                ],
                0),
            Question(
                '7. What does a market economy primarily rely on?',
                [
                  'A) Supply and demand',
                  'B) Government regulations',
                  'C) Fixed prices',
                  'D) Central planning'
                ],
                3),
            Question(
                '8. Which term refers to a situation where a company controls the entire supply chain?',
                [
                  'A) Vertical Integration',
                  'B) Horizontal Integration',
                  'C) Diversification',
                  'D) Conglomeration'
                ],
                1),
            Question(
                '9. What does the acronym CEO stand for?',
                [
                  'A) Chief Executive Officer',
                  'B) Chief Evaluation Officer',
                  'C) Chief Engineering Officer',
                  'D) Chief Economic Officer'
                ],
                2),
            Question(
                '10. What is the primary function of marketing?',
                [
                  'A) To promote and sell products or services',
                  'B) To manage company finances',
                  'C) To handle customer service',
                  'D) To design products'
                ],
                1),
          ],
        ),
        QuizSet(
          name: 'Intermediate',
          questions: [
            Question(
                '1. What does "economies of scale" refer to?',
                [
                  'A)Increased costs due to higher production ',
                  'B)  Reduction in per-unit cost due to increased production',
                  'C) Stable costs despite changes in production',
                  'D) Increased revenue from economies of scope'
                ],
                1),
            Question(
                '2. Which term describes the total value of a company’s outstanding shares of stock?',
                [
                  'A)  Price-to-Earnings Ratio ',
                  'B) Earnings Per Share',
                  'C) Dividend Yield',
                  'D)Market Capitalization'
                ],
                3),
            Question(
                '3. What does "monetary policy" primarily involve?',
                [
                  'A) Managing interest rates and money supply',
                  'B) Regulating trade policies',
                  'C) Controlling fiscal deficits',
                  'D) Setting up import/export tariffs'
                ],
                0),
            Question(
                '4. In what type of market structure does a single seller dominate and control prices?',
                [
                  'A)Perfect Competition  ',
                  'B) Oligopoly',
                  'C) Monopoly',
                  'D) Monopolistic Competition'
                ],
                2),
            Question(
                '5. What is the "liquidity ratio" used to measure?',
                [
                  'A) A company’s ability to pay off short-term obligations',
                  'B) A company’s profitability over a period',
                  'C) A company’s long-term debt levels',
                  'D) A company’s market share'
                ],
                0),
            Question(
                '6. What is "strategic management"?',
                [
                  'A) The process of defining the strategy and making decisions to achieve long-term goals',
                  'B) The day-to-day operations management of a company',
                  'C) The management of financial investments',
                  'D) The control of supply chain processes'
                ],
                0),
            Question(
                '7. Which financial metric is used to assess a company’s profitability relative to its equity?',
                [
                  'A) Return on Equity (ROE)',
                  'B) Return on Assets (ROA)',
                  'C) Earnings Before Interest and Taxes (EBIT)',
                  'D) Gross Margin'
                ],
                0),
            Question(
                '8. What is "corporate governance"?',
                [
                  'A)The management of day-to-day business operations   ',
                  'B)   The system of rules, practices, and processes by which a company is directed and controlled',
                  'C) The strategy for market expansion',
                  'D) The process of acquiring new businesses'
                ],
                1),
          ],
        ),
        QuizSet(
          name: 'Difficult',
          questions: [
            Question(
                '1. What is the primary purpose of "quantitative easing" by central banks?',
                [
                  'A) To decrease the national deb',
                  'B) To reduce government spending',
                  'C) t To increase money supply and lower interest rates',
                  'D) To increase tax rates'
                ],
                2),
            Question(
                '2. What is the "Gini coefficient" used to measure?',
                [
                  'A)  The efficiency of a company’s supply chain ',
                  'B) The stability of financial markets',
                  'C) Income inequality within a nation',
                  'D) The profitability of a business'
                ],
                2),
            Question(
                '3. In which financial statement would you most likely find "goodwill" listed?',
                [
                  'A)  Statement of Shareholders’ Equity',
                  'B) Income Statement',
                  'C) Cash Flow Statement',
                  'D)Balance Sheet '
                ],
                3),
            Question(
                '4. What does "forward guidance" refer to in monetary policy?',
                [
                  'A) Central banks’ communication about future policy intentions',
                  'B) The prediction of future market trends',
                  'C) The implementation of new fiscal policies',
                  'D) The management of trade relations with other countries'
                ],
                0),
            Question(
                '5. What is "Hedge Accounting" used for in financial reporting?',
                [
                  'A) To manage the risks associated with financial instruments',
                  'B) To increase the profitability of a company',
                  'C) To report income and expenses more accurately',
                  'D) To measure the efficiency of investment portfolios'
                ],
                0),
          ],
        ),
      ],
    ),
    Category(
      name: 'Cloud Computing',
      image: 'images/cloud computing.png',
      quizquestionsets: [
        QuizSet(
          name: 'Beginner',
          questions: [
            Question(
                '1. What is cloud computing?',
                [
                  'A) Computing using physical servers',
                  'B) Storing data on a local disk',
                  'C) Accessing computing resources over the internet',
                  'D) Programming using a desktop computer'
                ],
                2),
            Question(
                '2. Which of the following is a major cloud service provider?',
                [
                  'A) Microsoft Azure',
                  'B) Windows XP',
                  'C) Adobe Photoshop',
                  'D) Oracle Database'
                ],
                0),
            Question(
                '3. What does IaaS stand for in cloud computing?',
                [
                  'A) Internet as a Service',
                  'B) Infrastructure as a Service',
                  'C) Integration as a Service',
                  'D) Information as a Service'
                ],
                1),
            Question(
                '4. Which cloud model allows users to manage their own operating systems and applications?',
                ['A) SaaS', 'B) PaaS', 'C) IaaS', 'D) DaaS'],
                2),
            Question(
                '5. What does SaaS stand for?',
                [
                  'A) Software as a Service',
                  'B) Security as a Service',
                  'C) Server as a Service',
                  'D) Storage as a Service'
                ],
                0),
            Question(
                '6. Which of the following is an example of a SaaS application?',
                [
                  'A)IBM Cloud ',
                  'B) Amazon Web Services',
                  'C) Microsoft Azure',
                  'D)  Google Drive'
                ],
                3),
            Question(
                '7. What is the primary advantage of cloud computing?',
                [
                  'A) Unlimited local storage',
                  'B) Cost efficiency and scalability',
                  'C) Increased hardware maintenance',
                  'D) Fixed resource limits'
                ],
                1),
            Question(
                '8. Which cloud computing service model provides a platform for developers to build and deploy applications?',
                ['A) IaaS', 'B) PaaS', 'C) SaaS', 'D) CaaS'],
                1),
            Question(
                '9. What does PaaS stand for?',
                [
                  'A) Platform as a Service',
                  'B) Processor as a Service',
                  'C) Product as a Service',
                  'D) Programming as a Service'
                ],
                0),
            Question(
                '10. Which cloud deployment model is hosted and managed by a third-party provider?',
                [
                  'A) Private Cloud',
                  'B) Hybrid Cloud',
                  'C) Community Cloud',
                  'D) Public Cloud'
                ],
                3),
          ],
        ),
        QuizSet(
          name: 'Intermediate',
          questions: [
            Question(
                '1. What is a virtual private cloud (VPC)?',
                [
                  'A) A virtual machine within a public cloud',
                  'B) A secure, isolated section of a public cloud',
                  'C) A local data center network',
                  'D) A cloud service for private networking'
                ],
                1),
            Question(
                '2. Which of the following is a key benefit of cloud elasticity?',
                [
                  'A) Fixed resource allocation',
                  'B) Reduced network bandwidth ',
                  'C) Increased hardware costs',
                  'D) Ability to scale resources up or down as needed'
                ],
                3),
            Question(
                '3. What is a cloud-native application?',
                [
                  'A) An application designed to run on a traditional server',
                  'B) An application built specifically for cloud environments',
                  'C) An application that only runs on local machines',
                  'D) An application requiring manual scaling'
                ],
                1),
            Question(
                '4. What is the purpose of a load balancer in cloud computing?',
                [
                  'A) To distribute incoming network traffic across multiple servers ',
                  'B) To store data securely',
                  'C) To encrypt data at rest',
                  'D) To back up data regularly'
                ],
                0),
            Question(
                '5. What does the term “multi-tenancy” refer to in cloud computing?',
                [
                  'A) Multiple backup solutions for data protection',
                  'B) Multiple instances of the same application running',
                  'C) Multiple data centers for redundancy',
                  'D) Multiple users sharing the same physical resources securely'
                ],
                3),
            Question(
                '6. What is the purpose of a cloud service level agreement (SLA)?',
                [
                  'A) To configure network settings',
                  'B) To define the terms of service between the provider and the user',
                  'C) To specify hardware requirements',
                  'D) To manage user access permissions'
                ],
                1),
            Question(
                '7. What is a cloud “snapshot”?',
                [
                  'A) A backup of a system’s current state',
                  'B) A log of cloud service usage',
                  'C) A report of data analytics',
                  'D) A monitoring tool for performance'
                ],
                0),
            Question(
                '8. Which of the following is a common use case for serverless computing?',
                [
                  'A) Running high-performance databases',
                  'B) Hosting complex enterprise applications',
                  'C) Executing short-lived, event-driven functions',
                  'D) Managing long-term storage solutions'
                ],
                2),
            Question(
                '9. What is the function of cloud orchestration?',
                [
                  'A) Managing the provisioning and coordination of cloud resources',
                  'B) Encrypting data for security',
                  'C) Monitoring cloud service performance',
                  'D) Designing user interfaces for applications'
                ],
                0),
            Question(
                '10. Which of the following is an example of a container orchestration platform?',
                [
                  'A)Amazon RDS',
                  'B) Google Drive',
                  'C) Kubernetes ',
                  'D) Docker Compose'
                ],
                2),
          ],
        ),
      ],
    ),
    Category(
      name: 'Current Affair',
      image: 'images/current affairs.png',
      quizquestionsets: [
        QuizSet(
          name: ' Beginner',
          questions: [
            Question(
                '1. Who is the current President of the United States?',
                [
                  'A) George W. Bush',
                  'B) Donald Trump',
                  'C) Barack Obama',
                  'D)  Joe Biden'
                ],
                3),
            Question(
                '2. What is the name of the recent Mars rover sent by NASA?',
                [
                  'A) Curiosity',
                  'B) Opportunity',
                  'C) Perseverance',
                  'D) Spirit'
                ],
                2),
            Question('3. Which country recently hosted the Summer Olympics?',
                ['A) Japan', 'B) China', 'C) Brazil', 'D) United Kingdom'], 0),
            Question(
                '4. What major global event took place in 2020 that affected many countries?',
                [
                  'A) The COVID-19 pandemic',
                  'B) The United Nations General Assembly',
                  'C) The G20 Summit',
                  'D) The World Economic Forum'
                ],
                0),
            Question(
                '5. What significant climate agreement was recently renewed or signed?',
                [
                  'A) Geneva Convention ',
                  'B) Kyoto Protocol',
                  'C) Montreal Protocol',
                  'D) Paris Agreement '
                ],
                3),
            Question(
                '6. Which country recently became the first to approve a COVID-19 vaccine for children?',
                [
                  'A) United States',
                  'B) China',
                  'C) Russia',
                  'D) United Kingdom'
                ],
                1),
            Question(
                '7. What international organization recently launched a new climate action plan?',
                [
                  'A) World Health Organization',
                  'B) International Monetary Fund',
                  'C) United Nations',
                  'D) World Trade Organization'
                ],
                2),
            Question(
                '8. Who won the 2024 Nobel Peace Prize?',
                [
                  'A) Malala Yousafzai',
                  'B) Greta Thunberg',
                  'C) World Food Programme',
                  'D) International Committee of the Red Cross'
                ],
                2),
            Question(
                '9. What recent technological advancement has been made in the field of artificial intelligence?',
                [
                  'A) Quantum computing',
                  'B) Autonomous vehicles',
                  'C) General AI',
                  'D) Superintelligent AI'
                ],
                1),
            Question(
                '10. Which new trade agreement was signed between major countries recently?',
                [
                  'A) United States-Mexico-Canada Agreement',
                  'B) Trans-Pacific Partnership',
                  'C) Regional Comprehensive Economic Partnership',
                  'D) North American Free Trade Agreement'
                ],
                2),
          ],
        ),
        QuizSet(
          name: 'Intermediate',
          questions: [
            Question(
                '1. Which country recently announced plans to phase out coal power by 2030?',
                ['A) Germany', 'B) Australia', 'C) India', 'D) United States'],
                0),
            Question(
                '2. Who became the new Secretary-General of the United Nations in 2023?',
                [
                  'A)  Kofi Annan',
                  'B) Ban Ki-moon',
                  'C)  António Guterres',
                  'D) Amina J. Mohammed'
                ],
                2),
            Question(
                '3. What major space mission did NASA complete in 2023?',
                [
                  'A) Lunar Gateway',
                  'B) James Webb Space Telescope',
                  'C) Mars Sample Return',
                  'D) Europa Clipper'
                ],
                1),
            Question(
                '4. Which country recently hosted the COP27 climate summit?',
                [
                  'A) Egypt',
                  'B) Brazil',
                  'C) United Kingdom',
                  'D) United Arab Emirates'
                ],
                0),
            Question(
                '5. What significant economic policy did the Federal Reserve implement in 2023?',
                [
                  'A) Interest rate hikes',
                  'B) Quantitative easing',
                  'C) Cryptocurrency regulations',
                  'D) Universal basic income'
                ],
                0),
            Question(
                '6. What global health issue was highlighted by the World Health Organization in 2023?',
                [
                  'A) Antimicrobial resistance',
                  'B) COVID-19 variants',
                  'C) Mental health crisis',
                  'D) Vaccine distribution'
                ],
                2),
            Question(
                '7. Who won the 2023 Nobel Prize in Literature?',
                [
                  'A) Olga Tokarczuk',
                  'B) Annie Ernaux',
                  'C) Salman Rushdie',
                  'D) Haruki Murakami'
                ],
                1),
            Question(
                '8. What major geopolitical conflict was a focus in international news in 2023?',
                [
                  'A) Russia-Ukraine conflict',
                  'B) China-Taiwan tensions',
                  'C) North Korea-South Korea relations',
                  'D) Iran-Israel tensions'
                ],
                0),
            Question(
                '9. Which technology company launched a groundbreaking AI model in 2023?',
                ['A) Google', 'B) Microsoft', 'C) OpenAI', 'D) IBM'],
                2),
            Question(
                '10. What significant environmental agreement was signed in 2023?',
                [
                  'A) The Kigali Amendment',
                  'B) The Paris Agreement',
                  'C) The Montreal Protocol',
                  'D) The Glasgow Climate Pact'
                ],
                3),
          ],
        ),
      ],
    ),
    Category(
      name: 'Entertainment',
      image: 'images/entertainment.png',
      quizquestionsets: [
        QuizSet(
          name: 'Beginner',
          questions: [
            Question(
                '1. Who played the character of Jack Dawson in the movie "Titanic"?',
                [
                  'A) Leonardo DiCaprio',
                  'B) Brad Pitt',
                  'C) Tom Hanks',
                  'D) Johnny Depp'
                ],
                0),
            Question(
                '2. What is the name of the fictional town where "Stranger Things" is set?',
                [
                  'A) Springfield',
                  'B) Riverdale',
                  'C)  Hawkins',
                  'D) Stars Hollow'
                ],
                2),
            Question(
                '3. Who is the famous British singer known for hits like "Rolling in the Deep"?',
                [
                  'A) Adele',
                  'B) Ed Sheeran',
                  'C) Sam Smith',
                  'D) Harry Styles'
                ],
                0),
            Question(
                '4. In which movie would you find the character "Buzz Lightyear"?',
                [
                  'A)  Himla plung',
                  'B) Finding Nemo',
                  'C) Cars',
                  'D)Toy Story'
                ],
                3),
            Question(
                '5. Who is known as the "King of Pop"?',
                [
                  'A) Michael Jackson',
                  'B) Elvis Presley',
                  'C) Prince',
                  'D) David Bowie'
                ],
                0),
            Question(
                '6. What TV show features a group of friends living in New York City?',
                [
                  'A)How I Met Your Mother ',
                  'B)  Friends',
                  'C) The Office',
                  'D) Seinfeld'
                ],
                1),
            Question(
                '7. Which movie franchise features a character named "Harry Potter"?',
                [
                  'A)Percy Jackson ',
                  'B) The Chronicles of Narnia',
                  'C)  Harry Potter',
                  'D) Lord of the Rings'
                ],
                2),
            Question(
                '8. What is the name of the popular video game character created by Nintendo, known for jumping on mushrooms?',
                ['A) Rahino', 'B) Luigi', 'C) Link', 'D)   Mario'],
                3),
            Question(
                '9. Which singer is known for the hit song "Shape of You"?',
                [
                  'A) Shawn Mendes',
                  'B) Justin Bieber',
                  'C)   Ed Sheeran ',
                  'D) Bruno Mars'
                ],
                2),
            Question(
                '10. What is the name of the fictional wizarding school in "Harry Potter"?',
                [
                  'A) Hogwarts',
                  'B) Durmstrang',
                  'C) Beauxbatons',
                  'D) Ilvermorny'
                ],
                0),
          ],
        ),
        QuizSet(
          name: 'Intermediate',
          questions: [
            Question(
                '1. In which film did Robert De Niro play the character Travis Bickle?',
                [
                  'A)The Godfather ',
                  'B) Goodfellas',
                  'C)    Taxi Driver',
                  'D) Raging Bull'
                ],
                2),
            Question(
                '2. Which streaming service is known for producing the series "The Crown"?',
                ['A) Disney+', 'B) Hulu', 'C) Amazon Prime', 'D)  Netflix'],
                3),
            Question(
                '3. What is the title of the debut album released by Beyoncé as a solo artist?',
                [
                  'A) Dangerously in Love',
                  'B) B\'Day',
                  'C) I Am... Sasha Fierce',
                  'D) 4'
                ],
                0),
            Question(
                '4. Which actor won the Academy Award for Best Actor for his role in "There Will Be Blood"?',
                [
                  'A) Daniel Day-Lewis',
                  'B) Johnny Depp',
                  'C) Christian Bale',
                  'D) Tom Hanks'
                ],
                0),
            Question(
                '5. What is the name of the fictional country in the film "Black Panther"?',
                ['A) Zamunda', 'B)  Wakanda', 'C) Elbonia', 'D) Genosha'],
                1),
            Question(
                '6. Which TV show features a character named "Walter White" who turns to cooking methamphetamine?',
                ['A) Breaking Bad', 'B) Narcos', 'C) Ozark', 'D) The Sopranos'],
                0),
            Question(
                '7. What is the name of the fictional family in the animated series "The Simpsons"?',
                ['A) Simpson', 'B) Griffin', 'C) Belcher', 'D) Pritchett'],
                0),
            Question(
                '8. Who directed the film "Inception"?',
                [
                  'A)Quentin Tarantino  ',
                  'B) Steven Spielberg',
                  'C) Christopher Nolan',
                  'D) Martin Scorsese'
                ],
                2),
            Question(
                '9. What is the name of the host of "The Tonight Show" since 2014?',
                [
                  'A)Conan O\'Brien ',
                  'B) Jay Leno',
                  'C)  Jimmy Fallon',
                  'D) David Letterman'
                ],
                2),
            Question(
                '10. Which film series features a secret agent known as "James Bond"?',
                [
                  'A) James Bond',
                  'B) Mission: Impossible',
                  'C) Bourne',
                  'D) Spy Kids'
                ],
                0)
          ],
        ),
        QuizSet(
          name: 'Difficult',
          questions: [
            Question(
                '1. Who composed the original score for the film "The Good, the Bad and the Ugly"?',
                [
                  'A) Bernard Herrmann ',
                  'B) John Williams',
                  'C) Hans Zimmer',
                  'D) Ennio Morricone'
                ],
                3),
            Question(
                '2. In which film did Marlon Brando deliver the famous line, "I could have been a contender"?',
                [
                  'A) On the Waterfront',
                  'B) The Godfather',
                  'C) Apocalypse Now',
                  'D) A Streetcar Named Desire'
                ],
                0),
            Question(
                '3. What was the first feature film to be completely animated using CGI?',
                [
                  'A) Toy Story',
                  'B) Finding Nemo',
                  'C) Shrek',
                  'D) The Incredibles'
                ],
                0),
            Question(
                '4. Which novel won the Pulitzer Prize for Fiction in 2011?',
                [
                  'A) Beloved',
                  'B) The Help',
                  'C) The Road',
                  ' D) A Visit from the Goon Squad'
                ],
                3),
            Question(
                '5. Who played the character of "Hannibal Lecter" in the film "The Silence of the Lambs"?',
                [
                  'A) Anthony Hopkins',
                  'B) Jack Nicholson',
                  'C) Robert De Niro',
                  'D) Al Pacino'
                ],
                0),
            Question(
                '6. In the TV series "The Wire", what is the primary occupation of the character Omar Little?',
                [
                  'A) Drug Dealer',
                  'B) Police Officer',
                  'C) Private Investigator',
                  'D) Hitman'
                ],
                3),
            Question(
                '7. Who directed the film "Pulp Fiction"?',
                [
                  'A)  Steven Spielberg',
                  'B) Martin Scorsese',
                  'C) Francis Ford Coppola',
                  'D) Quentin Tarantino'
                ],
                3),
            Question(
                '8. What is the name of the first James Bond film released in 1962?',
                [
                  'A) From Russia with Love',
                  'B) Dr. No',
                  'C) Goldfinger',
                  'D) Thunderball'
                ],
                1),
            Question(
                '9. Which famous artist released the album "The Dark Side of the Moon" in 1973?',
                [
                  'A)The Rolling Stones ',
                  'B) Led Zeppelin',
                  'C)Pink Floyd ',
                  'D) David Bowie '
                ],
                2//right answerr
                ),
            Question(
                '10. In which film did Tilda Swinton play the character of "The White Witch"?',
                [
                  'A) Harry Potter and the Prisoner of Azkaban ',
                  'B) Snow White and the Huntsman',
                  'C) The Chronicles of Narnia: The Lion, the Witch and the Wardrobe',
                  'D) Stardust'
                ],
                2),
          ],
        ),
      ],
    ),
  ];
}
