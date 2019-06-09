# Natural-Language-Understanding-for-Dialog-Systems

Python program NLU.py that uses a supervised method to extract classid-s (e.g., EECS 280) and names (e.g., data structures and algorithms) from an annotated dataset of student statements provided in the NLU.train and NLU.test files. For each token in the input text, your classifier will have to assign an I(nside), O(utside), or B(eginning) label, to indicate if the token belongs to a class id or a class name.

To illustrate the IOB notation, consider the following examples:

The best class I took here was computer and network security.
<class
name=computer and network security link=Computer and Network Security taken=true
sentiment=positive>

This corresponds to the following IOB annotation:
The/O best/O class/O I/O took/O here/O was/O computer/B and/I network/I security/I ./O

I was in EECS 579 last semester.
<class id=579
department=EECS taken=true semester=last semester>

The IOB annotation:
I/O was/O in/O EECS/O 579/B last/O semester/O ./O
 

My favorite class was probably EECS 498, computer and network security.
<class id=498
department=EECS sentiment=positive
name=computer and network security link=Computer and Network Security taken=true>

The IOB annotation:
My/O favorite/O class/O was/O probably/O EECS/O 498/B ,/O computer/B and/I network/I security/I ./O

In order to generate an IOB notation for an example, after the text is tokenized, the values of the “id” and “name” fields found under the <class> tags are mapped onto the text. For each of the sequences found in the text, the first token in the sequence is labeled as B and the remaining tokens in the sequence are labeled as I. All the tokens outside the sequences are labeled as O.

Classifier will have to label each token as I, O, or B. You will train your classifier on the tokens extracted from the training data, and test the classifier on the tokens from the test data. For each token, you will have to extract the following five features:

➔ the value of the token
➔ is token all uppercase?
➔ does token start with capital?
➔ length of token
➔ does the token consist only of numbers?

Additionally, implement at least three other features of your choice.

If you want, you can use the eecs_dict that maps class numbers to class names, provided in dicts.py. The use of this dictionary is optional.

Notes:

➔ Ignore the <instructor> tags
➔ You should also ignore the texts that do not have a <class> tag associated with them

Programming guidelines:
Your program should perform the following steps:

❖	Transform the training and the test examples into the IOB notation, as described before.
 
❖	For each token, generate a feature vector consisting of the five features described before, plus at least three features of your choice.
❖	Train your system on the tokens from the training data and apply it on the tokens from the test data. Use a classifier of your choice from sk-learn.
❖	Compare the IOB tags predicted by your system on the test data against the correct (gold-standard) IOB tags and calculate the accuracy of your system.

The NLU.py program should be run using a command like this:
% python NLU.py NLU.train NLU.test

The program should produce at the standard output the accuracy of the system, as a percentage. It should also generate a file called NLU.test.out, which includes the textual examples in the test file along with the IOB tags predicted for each token.

