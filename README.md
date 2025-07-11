# 2025 Fall CPSCI-366

## Lecture Details

**Lecture time**: Tuesday and Thursday 1:00 PM - 2:15 PM

**Location**: Taylor Science Center, 3040 LEC

**Office Hours**: 

1. Tuesday 10:00 AM - Noon

2. Thursday 2:15 PM - 4:15 PM

3. or by appointment via email. 

You can meet me at my office (SCCT 2016) or via Zoom (if necessary). 

**Email**: schen3@hamilton.edu

**Gradescope**: https://www.gradescope.com Entry Code: 8KXJGY


## Course Description

This course discusses deep learning, covering foundational machine learning principles (loss functions, bias-variance trade-off, and optimization), advanced models (Convolutional Neural Networks, Recurrent Neural Networks, and Transformers), and their applications. The course is heavy-math and code instance and project-based, emphasizing applying these techniques to real-world datasets using the deep learning framework.

## Learning Objectives

In this class, we will learn:

1. Basic machine/deep learning concepts
2. Classic and state-of-the-art deep learning architectures
3. How to build, train, and test the deep learning model on real-world data using PyTorch

After this class, you should be able to:

1. Convert a real-life or research question into a computer science problem
2. Build a deep learning pipeline to solve the problem
3. Correctly evaluate and improve your deep learning model's performance

## Prerequisites

1. This class heavily relies on Python. You should be comfortable using Python for programming or be familiar with other programming languages and able to learn and use Python quickly.
2. Be familiar with the basic linear algebra and calculus is required.
3. Background in statistics is desried but not required. (We will cover the necessary part in class)

## Course Materials

### Textbook

**Required**: [Dive into Deep Learning](https://d2l.ai/) 

**Recommended**: [Deep Learning](https://www.deeplearningbook.org/)

### GPU Setup
Every four studetns will be assigned one Linux machine with GeForce RTX 5070 TI gpu for assighment and final project. **If you require more computation power, you could purchase some online services (This is not required).**


### Laptops and Electronics
You should not use a labtop, phone or any similar device during lectures. If you take notes on a tablet, then you should not be typing on it during class, but only writing (e.g. with a stylus) unless you require accommodation for a disability. Tablets should be kept flat on the desk and should not be propped up unless you require accommodation for a disability. If you would like to discuss this restriction, you are always welcome to come talk to me about it.

**You can only use your laptop  for coding at mini-lab time. You can't use the computer for anything unrelated to the course content. For example, I found twice that if you use the computer for something unrelated (E.g.: Facebook, YouTube, game,...) to the course, you will lose 5% of your final grade.** For example: If your final score is 90, because of using the laptop for unrelated to the course content twice. Your score will drop to 85.5.

## Academic Integrity & Collaboration

You should work on all assignments by yourself. You can discuss your assignments with me, your classmates, your friends, or other professors, but you are **NOT** allowed to use others' code.

You can work on the final project on your own or team up with up to 3 students. If you choose to work in a team, your team must report each member's contribution. I encourage all team members to join the entire project pipeline as much as possible, instead of each team member only being in charge of one part of the project. Here is an [example](https://github.com/XiaoChen1992/CPSCI-307/blob/main/project_example/contrbution.md). You should include this file in your own project, otherwise, all team members will lose 5% of the project score. For example, if your final project's score is 90, because of a missing file, the score will drop to 85.5.

### AI Policy

### Assignments

You are **NOT** allowed to use any generative AI tools (online or offline) for your assignments. Any code from generative AI tools (online or offline) will be considered cheating and you will receive 0 points for the assignments.

### Final Project

You are **ENCOURAGED** to use AI tools for your final project, but you **NEED** to provide your prompt history to me, **otherwise you will lose 5%** of your final project score. For example, if your final project's score is 90, because of missing prompt history, the score will drop to 85.5. Here is a prompt history [example](https://github.com/XiaoChen1992/CPSCI-307/blob/main/project_example/prompt_history_example.pdf).


**THIS POLICY APPLIES ONLY TO THIS CLASS!!!** For other classes, follow the academic integrity policy of that class or ask your professor.

## Grading

Your final score will be comprised of the following weighted components:

1. **Assignments (40%)**: Three assignments, each assignment contributes 10% of your final score. The assignments' goal is to let you build, train, and test different types of neural networks by PyTorch.

2. **Midterm (20%)**: Two midterm exams (close book).

3. **Final Project (40%)**: The final project covers:
   
   1. Proposal (0%): Create a plan for the final project, up to 1 pages ([IEEE - Manuscript Templates for Conference Proceedings](https://www.ieee.org/conferences/publishing/templates.html))  
   
   2. Data (10%): Report how you collected your own data for the final project. Include the following parts: a. Where is the data source? b. How did you collect the data? c. How did you preprocess the data? The report is up to 1 page (IEEE format). You also need to provide some data examples to me.
   
   3. Code (10%): Upload your project's code to the gradescope. Your code should be well-organized and easy to read. Check this [example](https://github.com/XiaoChen1992/CPSCI-307/tree/main/project_example) to organize your code. If you do not follow the example's style, you will lose 5% of your final project score. For example, if your final project's score is 90, because of missing prompt history, the score will drop to 85.5.
   
   4. Performance (10%):
      
      1. I should be able to directly run your `inference.py` file in the terminal to use your model. For example:
         
         ```shell
         python inference.py --data /example/data1
         ```
      
      2. You need to report your learning curve through [Weights & Biases](https://wandb.ai/site). 
      
      3. Your model's performance should be evaluated by suitable metrics and achieve reasonable performance.
         
         1. The learning curve should show your model was learning, with no obvious overfitting, underfitting pattern or other issues.
         
         2. The model's test data does not have a leaking issue. 
   
   5. Presentation and report(10%): The report is up to 4 pages (IEEE format). A good presentation and report should cover the following parts:
         1. Introduction
         2. Method and matrerials
         3. Experiment and results
         4. Colution

You need to attend your final presentation **on time**, otherwise, you will loose whole final project's credits.

4. **Participation**: If you have three unexcused absences. You will lose 5% of your final score. For example, if your final score is 90, but you missed three classes, then your final score is 85.5. **Four unexcused absences will fail you to this class.** Check section **Attendance and Late Policy** for more details. 

Your final score will be a weighted sum of all three parts. The final score will convert to a letter grade by following the table:

| Score | Letter Grade |
| ----- | ------------ |
| >=93  | A            |
| 90~93 | A-           |
| 87~89 | B+           |
| 84~86 | B            |
| 80~83 | B-           |
| 77~79 | C+           |
| 74~76 | C            |
| 70~73 | C-           |
| 67~69 | D+           |
| 64~66 | D            |
| 60~63 | D-           |
| <60   | F            |

## Attendance and Late Work Policy

### Attendance

You are expected to attend every class. You may be excused only for college-sanctioned activities, and you must let me know about such absences as soon as you are notified. This includes missing class for religious, athletic, or academic conflicts. If you are sick or have an important appointment at the health or counseling center, please email me **before** the class and take care of yourself. If you must miss a class for a college-sanctioned activity, you must notify me prior to the class in question via email.

### Late Work

No late work will be accepted without prior permission. If you contact me at least one work day before the due date (unless faced with an emergency) with appropriate requests for an extension and/or makeup assignments, you will be given additional time to make up late assignments equal to the time lost due to unforeseen circumstances.

**Any unexcused late ness (within 24 hours) will result 10% score deduction for the assignment. For example, your assignment's score is 89, because of the late, the score will drop to $$89 \times 0.9 = 80.1$$. If you submit your assignment late more than 24 hours, I will not accept it, the score will be 0.**

## Seeking Help

### Accommodations

If you believe you may need accommodation for a disability, contact me privately within the first two weeks (2024-10-09) of the semester to discuss your specific needs. If you have not already done so, please contact Allen Harrison, Assistant Dean of Students for International Students and Accessibility, at 315-859-4021, or via email at aharriso@hamilton.edu. He is responsible for determining reasonable and appropriate accommodations for students with disabilities on a case-by-case basis. 

### Mental Health and Wellness

If you are feeling isolated, depressed, sad, anxious, angry, or overwhelmed, you aren’t alone: we all struggle sometimes. Don’t stay silent! Talk to a trusted confidant: a friend, a family member, a professor you trust. The counseling center offers completely confidential and highly professional services and can be contacted at 315-859-4340. If this seems like a difficult step, contact me. We can talk and call or walk to the counseling center together.

## Course Schedule

The course calendar is intended to be flexible enough to accommodate our class’s particular interests and needs while providing overall guidance and structure. The instructor may adjust topics/assignments as the course progresses.

| Week                           | Learning topics                                                                                                                                 | Student activities                                                                                                                                                                                                                                        |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2024-08-26 (Monday)<br>Week1   |                                                                                                                                                 |                                                                                                                                                                                                                                                           |
| 2024-09-02 (Monday)<br>Week 2  | 1. Introduction<br>2. Linear algebra + calculus<br>3. Linear regression<br>4. Basic optimization                                                | **Todo**: Work on Assignment 0                                                                                                                                                                                                                            |
| 2024-09-09 (Monday)<br>Week 3  | 1. Linear classification<br>2. loss function<br>3. softmax<br>4. MLP<br>5. activation function<br>6. Bais and variance trade-off, model fitting | **Due**: Assignment 0, bring it to class (2024-09-09)                                                                                                                                                                                                     |
| 2024-09-16 (Monday)<br>Week 4  | 1. Weight decay<br>2. drop out<br>3. Stochastic Gradient Descent<br>4. Momentum, AdaGrad, Adam<br>5. Learning rate schedules                    | **Todo**:1. Work on Assignment 1 (2024-09-16)<br>2. Start to work on the final project's proposal                                                                                                                                                         |
| 2024-09-23 (Monday)<br>Week 5  | 1. Initialization<br>2. Build our first model on House Prices data<br>3. Convolution layer/ Dilated Convolution layer                           | **Due**: Discuss the final project proposal with me at any office hour this week.                                                                                                                                                                         |
| 2024-09-30 (Monday)<br>Week 6  | 1. Polling layer <br>2. AlexNet<br>3. VGG<br>4. NiN<br>5. GoogleNet                                                                             | **Due**: Submit assignment 1 to gradescope before 2024-09-30 2:30 PM<br>**Todo**: Start to work on data plan                                                                                                                                              |
| 2024-10-07 (Monday)<br>Week 7  | 1. Batch Normalization<br>2. More drop out<br>3. ResNet                                                                                         | **Due**: Submit the final project proposal to gradescope before 2024-10-07 2:30 PM <br>**Todo**: 1. Discuss the data plan with me at any office hour this week. <br>2. Work on Assignment 2 (2024-10-07)                                                  |
| 2024-10-14 (Monday)<br>Week 8  | 1. Build a CNN for image classification task<br>2. pytorch lightning<br>3. Weights & Biases                                                     | Midterm: 2024-10-14 (Monday),  7:00 PM - 8:30 PM                                                                                                                                                                                                       |
| 2024-10-21 (Monday)<br>Week 9  | Guest Talk<br>1. Data augmentation<br>2. Fine-tune                                                                                              | **Due**: 1. Submit data collection report to gradescope before 2024-10-21 2:30 PM<br>2. Submit Assignment 2 to gradescope before 2024-10-23 2:30 PM<br>**Todo**: 1. Start to work on model building and training<br> 2. Work on Assignment 3 (2014-10-23) |
| 2024-10-28 (Monday)<br>Week 10 | Object Detection and Image Segmentation                                                                                                         |                                                                                                                                                                                                                                                           |
| 2024-11-04 (Monday)<br>Week 11 | 1. Review probability<br>2. Word vector and embeddings<br>3. Sequential model                                                                   | **Due**: 1. Discuss training details with me at any office hour this week.<br>2. Submit Assignment 3 to gradescope before 2024-11-16 2:30 PM                                                                                                              |
| 2024-11-11 (Monday)<br>Week 12 | 1. GRU<br>2. LSTM/bi-LSTM<br>3.Seq2seq                                                                                                          |                                                                                                                                                                                                                                                           |
| 2024-11-18 (Monday)<br>Week 13 | 1. Word embedding<br>2. Attention, axial attention, channel attention                                                                           | **Todo**: Start to work on poster                                                                                                                                                                                                                         |
| Thanksgiving                   | No Class                                                                                                                                        |                                                                                                                                                                                                                                                           |
| 2024-12-02 (Monday)<br>Week 15 | 1. self-attention<br>2. Transformer<br>3. ViT (vision transformer)<br>                                                                          | **Due**: Submit poster to gradescope before  2024-12-04 2:30 PM                                                                                                                                                                                           |
| 2024-12-09 (Monday)<br>Week 16 | Introduction to diffusion                                                                                                                       |                                                                                                                                                                                                                                                           |
| 2024-12-13 (Friday)            |                                                                                                                                                 | **Due**: Submit project folder (includes all required codes, files, plots, etc.) to gradescope before 2024-12-13 5:00 PM                                                                                                                                  |
