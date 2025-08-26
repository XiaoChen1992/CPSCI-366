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

**Edstem**: https://edstem.org/us/join/RsCeD4


## Course Description

This course discusses deep learning, covering foundational machine learning principles (loss functions, bias-variance trade-off, and optimization), advanced models (Multilayer Perceptrons, Convolutional Neural Networks, Recurrent Neural Networks, and Transformers), and their applications. The course is heavy-math, code instance and project-based, emphasizing applying these techniques to real-world datasets using the deep learning framework.

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

**You can only use your laptop  for coding at mini-lab time. You can't use the computer for anything unrelated to the course content. If I found twice that if you use the computer for something unrelated (E.g.: Facebook, YouTube, game,...) to the course, you will lose 5% of your final grade.** For example, if your final score is 90, because of using the laptop for unrelated to the course content twice. Your score will drop to 85.5.

## Academic Integrity & Collaboration

You should work on all assignments by yourself. You can discuss your assignments with me, your classmates, your friends, or other professors, but you are **NOT** allowed to use others' code.

You can work on the final project on your own or team up with up to 4 students. If you choose to work in a team, your team must report each member's contribution. I encourage all team members to join the entire project pipeline as much as possible, instead of each team member only being in charge of one part of the project. Here is an [example](https://github.com/XiaoChen1992/CPSCI-366/blob/main/project_example/contrbution.md). You should include this file in your own project, otherwise, all team members will lose 5% of the project score. For example, if your final project's score is 90, because of a missing file, the score will drop to 85.5.

## AI Policy

### Assignments

You are **NOT** allowed to use any generative AI tools (online or offline) for your assignments. Any code from generative AI tools (online or offline) will be considered cheating and you will receive 0 points for the assignments and report it to College.

### Final Project

You are **ENCOURAGED** to use AI tools for your final project, but you **NEED** to provide your prompt history to me, **otherwise you will lose 5%** of your final project score. For example, if your final project's score is 90, because of missing prompt history, the score will drop to 85.5. Here is a prompt history [example](https://github.com/XiaoChen1992/CPSCI-366/blob/main/project_example/prompt_history_example.pdf).

### AI tools
Here some some free AI tools:
1. [Google Gimini](https://www.google.com/aclk?sa=L&ai=DChsSEwiXusXju6ePAxVnS0cBHRo9IE0YACICCAEQABoCcXU&ae=2&co=1&ase=2&gclid=CjwKCAjwk7DFBhBAEiwAeYbJsQ1bJ0oL0-WtFnrZh4HjgVtbuS1x-57QgkXXHb0_3H6J6pNNORyIURoCO8sQAvD_BwE&cce=2&category=acrcp_v1_71&sig=AOD64_15cEvKAT8mbgz8ZrE2D2E9DRizNg&q&nis=4&adurl&ved=2ahUKEwizv73ju6ePAxWbKlkFHecIKNkQ0Qx6BAgKEAE), student plan (100% free)
2. [Github Copilot Pro](https://github.com/education/students). 100% free.

I require everyone uses these two paid version AI tools for this class.

## Grading

Your final score will be comprised of the following weighted components:

1. **Assignments (30%)**: Three assignments, each assignment contributes 10% of your final score. The assignments' goal is to let you build, train, and test different types of neural networks by PyTorch.

2. **Midterm (25%)**: Two midterm exams (close book).

3. **Final Project (40%)**: The final project covers:
   
   1. Proposal (0%): Create a plan for the final project, up to 1 pages ([IEEE - Manuscript Templates for Conference Proceedings](https://www.ieee.org/conferences/publishing/templates.html))  
   
   2. Data (10%): Report how you collected your own data for the final project. Include the following parts: a. Where is the data source? b. How did you collect the data? c. How did you preprocess the data? The report is up to 2 page (IEEE format). You also need to provide some data examples to me.
   
   3. Code (10%): Upload your project's code to the gradescope. Your code should be well-organized and easy to read. Check this [example](https://github.com/XiaoChen1992/CPSCI-366/tree/main/project_example) to organize your code. If you do not follow the example's style, you will lose 5% of your final project score. For example, if your final project's score is 90, because of missing prompt history, the score will drop to 85.5.
   
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

      You need to attend your final presentation **on time**, otherwise, you get 0 as your final project score.
4. **Reading(5%)**: I will ask few questions at the begining of every class, to get full credits, you need to answer three questions with reasonable answers.

5. **Participation**: If you have three unexcused absences. You will lose 5% of your final score. **Four unexcused absences will fail you to this class. Also I will report it to College.** Check section **Attendance and Late Policy** for more details. 


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

**Any unexcused late ness (within 24 hours) will result 10% score deduction for the assignment. For example, your assignment's score is 89, because of the late, the score will drop to $89 \times 0.9 = 80.1$. If you submit your assignment late more than 24 hours, I will not accept it, the score will be 0.**

## Seeking Help

### Accommodations

If you believe you may need accommodation for a disability, contact me privately within the first two weeks (2025-09-12) of the semester to discuss your specific needs. If you have not already done so, please contact Allen Harrison, Assistant Dean of Students for International Students and Accessibility, at 315-859-4021, or via email at aharriso@hamilton.edu. He is responsible for determining reasonable and appropriate accommodations for students with disabilities on a case-by-case basis. 

### Mental Health and Wellness

If you are feeling isolated, depressed, sad, anxious, angry, or overwhelmed, you aren’t alone: we all struggle sometimes. Don’t stay silent! Talk to a trusted confidant: a friend, a family member, a professor you trust. The counseling center offers completely confidential and highly professional services and can be contacted at 315-859-4340. If this seems like a difficult step, contact me. We can talk and call or walk to the counseling center together.

## Course Schedule

The course [calendar](https://docs.google.com/spreadsheets/d/1zGZN_DY2zIlZTHcVgnipyDB5NmLKJd4CaQBFUWdWq3Y/edit?usp=sharing) is intended to be flexible enough to accommodate our class’s particular interests and needs while providing overall guidance and structure. The instructor may adjust topics/assignments as the course progresses.
