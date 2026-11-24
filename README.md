# **Lesson: Bayesian Algorithm in Machine Learning**




# **PART 1: Introduction to Machine Learning**

## **1. What is Machine Learning?**

Machine Learning (ML) is a branch of Artificial Intelligence (AI) that allows computers to learn from data and make predictions or decisions without being explicitly programmed for every situation.

Let's use an example.

Imagine you want a computer to recognize cats.

[Portraits of cats, collage with domestic animals in different situations](https://images.openai.com/static-rsc-4/gyq1kiSMkQ4QHtgcYbL1oS17T5IfDSB8AQggQBkY1ai-xG1hT6oNmBHZK6vZLSW8XHoHWQMS2i2wvDO9C1FJd63a1tlj5L-y1lpFc0Pn4VwFyMAflniin7RZ9OmwrDAuhnRJ2n32_HYOBVJLdBJQfhmnpQBPrLnay7XxMb4aXi5mxaPQ18U5kR6kd-V7clTi?purpose=inline)

[روزی برای وفادارترین دوست انسان‌ها | روز ملی سگ](https://images.openai.com/static-rsc-4/PvH09Tp69dCAtVcbFC4XOP61-FSNuBN7C3j56_oN1OTBFyGpTJCtExcp6doGVC2IWlCdMqJbpR-BZ3mhlpDF1COMANyMmJafpQo24mfgxX1mA4Z1H-cA4rhqg2yWFevb7T8WwiE8TXq3eBDbtjgvYd-xEG2NDyQL3mdfacrGO-M?purpose=inline)

You give the computer 1,000 pictures:

- 500 pictures labelled CAT.
- 500 pictures labelled DOG.

The computer studies these pictures and learns patterns that help distinguish cats from dogs.

When you give it a new picture, it predicts whether the picture contains a cat or a dog.

That's Machine Learning!

**Important:** The computer learns patterns from examples instead of relying only on manually written rules.

## **2. How does Machine Learning work?**

Your notes identify four steps.

1. **Input Data**

   Give the computer information, such as pictures, numbers or text.
2. **Training**

   The algorithm studies the information and identifies patterns.
3. **Model Building**

   The computer creates a mathematical model representing the patterns it learned.
4. **Prediction/Decision**

   The trained model uses what it learned to predict outcomes for new data.

For example, when predicting house prices:

| Input           | Output                |
| --------------- | --------------------- |
| House size      |                       |
| Location        | Predicted house price |
| Number of rooms |                       |

The model learns from previous house sales and uses those patterns to estimate prices for new houses.

## **3. Types of Machine Learning**

Your PDF identifies four categories.

### **A. Supervised Learning**

Learns from labelled data

Supervised learning means the computer learns using examples that already have correct answers.

Example:

| Picture          | Label |
| ---------------- | ----- |
| Picture of a cat | Cat   |
| Picture of a dog | Dog   |
| Picture of a cat | Cat   |

The labels tell the computer what each picture represents.

**Real-world example:** Predicting house prices using previous sales.

Algorithms in your notes:

- Linear Regression
- Decision Trees
- Neural Networks

### **B. Unsupervised Learning**

Learns from unlabelled data

Here, the computer receives data without predefined answers.

It must discover patterns or groups by itself.

[Unsupervised Machine Learning | by Chandan Pal | Feb, 2026 | Medium](https://images.openai.com/static-rsc-4/-PQH5ycldwSaCRvKNxFO1EF_5JqjxrVNVFliOrxl_Y2akCgCwkmhQ6Gk2Pvhc2oNxbEyNJTs2cC9KEuGXbX4ljjiTDWIkz613eZ9UsC9dlhsBEzzTdPnRICRxc2wwaUWWV-UPZ_36SaffrBbYPN3Mp44hc5yONelEyFPVmHzQlg?purpose=inline)

[medium.com](https://medium.com/%40techie_chandan/unsupervised-machine-learning-explained-clustering-pca-autoencoders-real-world-examples-72cc76f06491?utm_source=chatgpt.com)

Imagine giving a computer information about 1,000 customers without telling it which customers belong together.

The computer may discover groups such as:

- Customers who buy frequently.
- Customers who spend little.
- Customers who buy expensive products.

Algorithms in your notes:

- K-Means Clustering
- PCA
- Apriori

### **C. Reinforcement Learning**

Learns through rewards and penalties

This is learning through trial and error.

Imagine a robot learning to navigate a maze.

[AI and ML](https://images.openai.com/static-rsc-4/FjMzDxLY3XXaAdKNkISmWIIbGZ0pLSx085_HjBvIoGPyqNT1bPvWcz7xKXIq2Ou8Y79Hk4imM6wKQumNfENHZAFvoew6E5X5Mk-l8AWO-Cd4tZCTxOTlh7Ip7ZTPOf4E51Ue5fyqZucNQMhXsJlxLYFVga6PtTcFAGHDGL_gc6c?purpose=inline)

[pmexpertinc.com](https://www.pmexpertinc.com/digital-transformation2/?utm_source=chatgpt.com)

- The robot is the **Agent**.
- Moving left or right is an **Action**.
- Reaching the destination gives a **Reward**.
- The robot learns which actions produce better results.

Your notes give these algorithms:

- Q-Learning
- Deep Q-Networks (DQN)

### **D. Semi-Supervised and Self-Supervised Learning**

Your notes introduce these as approaches that learn using combinations of labelled and unlabelled information, or by creating learning signals from the data itself.

## **4. Real-world applications**

| Industry   | Application                   |
| ---------- | ----------------------------- |
| Healthcare | Disease detection from X-rays |
| Finance    | Fraud detection               |
| Retail     | Product recommendations       |
| Transport  | Self-driving cars             |
| NLP        | Chatbots and translation      |

**Remember this distinction:**

- Supervised = learns with answers.
- Unsupervised = finds patterns without answers.
- Reinforcement = learns through rewards and penalties.

# **PART 2: Understanding Bayes' Theorem**

This is the most important foundation of the topic.

## **5. What is Bayes' Theorem?**

Bayes' Theorem is a mathematical formula that helps us update our beliefs when new evidence becomes available.

Let's use a simple example.

Imagine you are trying to determine whether an email is spam.

Before reading the email, you estimate that 30% of emails are spam.

Then you notice that the email contains the word "FREE".

That new information changes how likely you think the email is to be spam.

Bayes' Theorem helps calculate that updated probability.

**In simple words: Bayes' Theorem uses previous knowledge and new evidence to calculate an updated probability.**

## **6. The four important terms in Bayes' Theorem**

Before learning the formula, understand these four words.

| Term       | Meaning                                           | Simple example                              |
| ---------- | ------------------------------------------------- | ------------------------------------------- |
| Prior      | Initial belief                                    | 30% of emails are spam                      |
| Likelihood | Probability of seeing evidence given a hypothesis | How likely is the word FREE in spam emails? |
| Evidence   | Overall probability of observing the evidence     | How often does FREE appear in all emails?   |
| Posterior  | Updated probability after seeing evidence         | Updated probability that the email is spam  |

The most important word is **Posterior**.

It is the final probability we calculate after considering new information.

## **7. Bayes' Theorem formula**

# **Bayes' Theorem**

$$[ P(A\mid B)=\frac{P(B\mid A)\times P(A)}{P(B)} \]$$

- $$(P(A\mid B)\)$$ = Posterior
- $$(P(B\mid A)\)$$ = Likelihood
- $$(P(A)\)$$ = Prior
- $$(P(B)\)$$ = Evidence

Let's understand the letters:

- A = The hypothesis we want to investigate.
- B = The evidence we have observed.

For example:

- A = An email is spam.
- B = The email contains the word FREE.

We want to calculate:

$$[ P(Spam\mid FREE) \]$$

This means: What is the probability that an email is spam, given that it contains FREE?

## **8. Understanding the calculation**

Your notes on page 7 introduce the formula, while page 12 gives a weather example.

Let's use that example.

We have the following information:

- Probability of playing when it is sunny: 3/9.
- Probability of sunny weather: 5/14.
- Probability of playing: 9/14.

We want to calculate:

$$[ P(Yes\mid Sunny) \]$$

In other words:

**What is the probability that someone will play when the weather is sunny?**

Step 1: Write the formula.

$$[ P(Yes\mid Sunny) = \frac{P(Sunny\mid Yes)\times P(Yes)} {P(Sunny)} \]$$

Step 2: Substitute the values.

$$[ =\frac{0.33\times0.64}{0.36} \]$$

Step 3: Multiply the numerator.

$$[ =\frac{0.2112}{0.36} \]$$

Step 4: Divide.

$$[ =0.5867\approx58.7\\% \]$$

The PDF rounds intermediate values differently and reports approximately 60%.

Answer: approximately 60%

**Interpretation:** Based on the given data, the estimated probability of playing when it is sunny is approximately 60%.

# **PART 3: Naive Bayes Algorithm**

## **9. What is Naive Bayes?**

Naive Bayes is a machine learning classification algorithm based on Bayes' Theorem.

It predicts which category something belongs to by calculating probabilities.

Examples:

- Is an email spam or not spam?
- Will it rain or not rain?
- Is a review positive or negative?
- Does a picture belong to a particular category?

Why is it called *Naive*?

Because it makes a simplifying assumption:

**Features are conditionally independent of one another given the class.**

For example, when classifying an email, it treats the presence of different words as independent pieces of evidence, once the class (spam or not spam) is known.

## **10. Three types of Naive Bayes**

Your PDF explains three important types.

[Descriptive and Inferential Statistics (Chapter 22) - The Cambridge Handbook of Research Methods and Statistics for the Social and Behavioral Sciences](https://images.openai.com/static-rsc-4/tTI96b-c9kcbvJ74eVIP4XUNwW6uY250aZOA8QqRM0Uo1TTrgIPUkoNQuim4wfcvyZSm3BBYAYcJitXvICj5JVyo7Xmnl-FQE_o-eJ4bsWe6Kz2a6KS4rFbjK4juv1a41oN7youC7M4UGao10-ugVx48kDKxQE20HNY-w_RnG0k?purpose=inline)

**A. Gaussian Naive Bayes**

Used when features are continuous numerical values.

Example: Temperature, blood pressure and other numerical measurements.

It assumes features follow a normal (bell-shaped) distribution.

[Multinomial Naive Bayes Classifier | by Yoann Mocquin | TDS Archive | Medium](https://images.openai.com/static-rsc-4/k8aXcnHMJHIMOlz05H5Y_9dGY19ym83_JfhXd-vczX8-jejutiRMCk7U6UbJ5_ksQrXFEhZuT3cPQEOYdTOn4hY7vfQlsOM1OY6B4ZPtpuSHHW1tWXUChmcXm4SJ6H0N1_LnRv1VaGykZ9PydZrkUwosmfNB6ZIQchSeCyuAu8k?purpose=inline)

**B. Multinomial Naive Bayes**

Used when features represent counts.

Example: An email contains:

- FREE = 4 times
- MONEY = 3 times

Commonly used in text classification.

[Encoding Categorical Data | TDS Archive](https://images.openai.com/static-rsc-4/RghjVwAicSGqvI7KBk-X8MtyDOwMDmNtZT5K2SRoFEJ2pQ_X7nOB2xbGTcXxihffqmJGSLHX1ymBElXU_-JaZvp085XUbwbV5OnZVWivKTBQ0AW7J5FGTAIELgGx7IfUTrR0H7-ykRe-sB8WucvToLJ5cATRo4hXa3vNrSr4OeU?purpose=inline)

**C. Bernoulli Naive Bayes**

Used when features are binary (0 or 1).

Example:

- Word WIN is present = 1.
- Word WIN is absent = 0.

### **How do you distinguish them in an exam?**

| Type        | Data               | Example        |
| ----------- | ------------------ | -------------- |
| Gaussian    | Continuous numbers | Temperature    |
| Multinomial | Counts             | Word frequency |
| Bernoulli   | Binary values      | Yes/No, 0/1    |

# **PART 4: Training, Testing and Validation**

## **11. What is a training dataset?**

Think of a student preparing for an examination.

The student uses textbooks and practice questions to learn.

In machine learning, the training dataset is the information used to teach the model.

Example:

You give a model 700 pictures of cats and dogs.

It studies those pictures and learns how to distinguish the two animals.

## **12. What is a testing dataset?**

Testing is like the final examination.

You give the model new pictures it has never seen before.

You then check whether it can correctly identify cats and dogs.

## **13. What is a validation dataset?**

Validation is like a practice examination during learning.

It helps you evaluate and adjust the model before its final testing.

Your PDF explains these three datasets on pages 13–14.

| Dataset    | Purpose                    |
| ---------- | -------------------------- |
| Training   | Teach the model            |
| Validation | Tune and improve the model |
| Testing    | Evaluate final performance |

## **14. Underfitting and Overfitting**

These are very important examination topics.

[What is Underfitting? How to Detect and Overcome High Bias in ML Models | DataCamp](https://images.openai.com/static-rsc-4/R5-gJh6MDSuA1kTu_r6e0E6L3Nn4p7MPR8p8uMilVkmJbdOXW5s3xfMHZNbs2XKwl9PuHNb8W0V76bhd2CayfvFM-2-J6F034JcMicsHvZ6aCoq88TpybRaOta8JOqysRVx22CC-BRX13MY6vFjFawb5nRpaB3uLTvAFtqzzLAY?purpose=inline)

**Underfitting**

Too simple

The model fails to learn important patterns. It performs poorly on training and testing data.

[Understanding Overfitting, Underfitting, Bias, and Variance](https://images.openai.com/static-rsc-4/bBmJJB2mwayWoeNHayF1bJpC9Jzkt7dHvMiH5UWhs0RSQ-403XogeV4gcn_xNufH1UGdtxO1J8hQqDoWftHoGBDamhoVUyY-ly4VnzX-eI2vU1RkIMZ9BNL3PzcSJIwH-fFnxJ3zMcQHAqeJw84HcYzo1rsm27bjXioSVeGFLJc?purpose=inline)

**Overfitting**

Too complex

The model memorizes training examples, including noise, and struggles with new data.

### **Good fit**

A good model learns meaningful patterns rather than memorizing examples.

| Model        | Training score | Testing score |
| ------------ | -------------- | ------------- |
| Underfitting | 60%            | 55%           |
| Overfitting  | 99%            | 65%           |
| Good fit     | 90%            | 88%           |

These are illustrative figures from your notes.

**How do we fix them?**

- Underfitting: Use a more complex model.
- Overfitting: Use regularization, more data or cross-validation.

# **PART 5: Evaluating a Machine Learning Model**

## **15. What is a confusion matrix?**

A confusion matrix is a table that compares the actual answers with the predictions made by a classification model.

Your PDF uses spam detection as an example.

Imagine a model checks 100 emails.

**Confusion matrix**

|                     | **Predicted Spam** | **Predicted Not Spam** |
| ------------------- | ------------------ | ---------------------- |
| **Actual Spam**     | 30 TP              | 10 FN                  |
| **Actual Not Spam** | 5 FP               | 55 TN                  |

Understand these four terms:

- **TP (True Positive):** Correctly predicted spam.
- **TN (True Negative):** Correctly predicted not spam.
- **FP (False Positive):** Predicted spam, but actually not spam.
- **FN (False Negative):** Predicted not spam, but actually spam.

## **16. Four evaluation metrics**

### **A. Accuracy**

Measures the percentage of all predictions that were correct.

$$[ Accuracy=\frac{TP+TN}{TP+TN+FP+FN} \]$$

Using our example:

$$[ =\frac{30+55}{100} \]$$

$$[ =\boxed{85\\%} \]$$

### **B. Precision**

Measures how many predicted positives were actually positive.

$$[ Precision=\frac{TP}{TP+FP} \]$$

$$[ =\frac{30}{30+5} =\boxed{85.71\\%} \]$$

### **C. Recall**

Measures how many actual positives the model successfully identified.

$$[ Recall=\frac{TP}{TP+FN} \]$$

$$[ =\frac{30}{30+10} =\boxed{75\\%} \]$$

### **D. F1-Score**

Combines precision and recall into one measure.

$$[ F1=2\times\frac{Precision\times Recall}{Precision+Recall} \]$$

Using our results:

$$[ =2\times\frac{0.8571\times0.75}{0.8571+0.75} \]$$

$$[ \boxed{F1\approx80\\%} \]$$

### **Quick revision table**

| Metric    | What it measures                     |
| --------- | ------------------------------------ |
| Accuracy  | Overall correctness                  |
| Precision | Reliability of positive predictions  |
| Recall    | Ability to find actual positives     |
| F1-Score  | Balance between precision and recall |

# **PART 6: Practical Python Project**

Your final section, pages 22–30, demonstrates how to build a weather prediction model using Gaussian Naive Bayes.

## **17. What are we building?**

[आजको मौसम पूर्वानुमान : पहाडी भू-भागमा आंशिक बदली हुने -  Setomasi](https://images.openai.com/static-rsc-4/6Tvmnpn7sfOvLnLiyWr9Sab4gKKQfBPVyXqaTqaJhDLnuNavHDPniWHKUpgpOPkxf_su6l8QFOQYJUeNRjRGasO8sPbobXWE7bSM0V3he4mYY6-Juz7xljZLowJ3c-SJuByqBlh-zR2mNBl8q-Z1Jf8pp0D5Q52OxRT-HhknIOg?purpose=inline)

[setomasi.com](https://setomasi.com/content/26311?utm_source=chatgpt.com)

We want a computer to predict whether it will rain based on:

1. Temperature
2. Humidity
3. Atmospheric pressure
4. Wind speed

The target variable is `Rain`.

- 0 = No Rain
- 1 = Rain

## **18. Understanding the Python workflow**

Your PDF uses these libraries:

| Library      | Purpose                |
| ------------ | ---------------------- |
| Pandas       | Organizing data        |
| NumPy        | Numerical calculations |
| Scikit-learn | Machine learning       |
| Matplotlib   | Visualizations         |
| Seaborn      | Statistical charts     |

The process is:

**1. Create weather dataset**

1,000 weather records

**2. Split the data**

700 training records + 300 testing records

**3. Train Gaussian Naive Bayes**

**4. Make predictions**

**5. Evaluate performance**

# **83.33%**

Test accuracy reported in your PDF

The PDF reports that the model achieved 83.33% accuracy.

However, it performed better at predicting No Rain than Rain:

- No Rain recall: 95%
- Rain recall: 46%

This means that even though overall accuracy was good, the model missed many actual rainy cases.

That is why we need more than accuracy when evaluating machine learning models.



**1. Which type of machine learning learns using labelled data?**

Unsupervised learning

Supervised learning

Reinforcement learning



Supervised learning learns from examples that already have correct labels.

**2. What does the posterior probability represent?**

Initial belief

Probability of the evidence

Updated belief after seeing evidence



Posterior is the updated probability after considering new evidence.

**3. Which Naive Bayes type is suitable for word counts?**

Gaussian

Multinomial

Bernoulli



Multinomial Naive Bayes works with count-based features such as word frequencies.

**4. What is overfitting?**

The model is too simple

The model memorizes training data and struggles with new data

The model has no training data



Overfitting happens when a model learns the training examples too closely, including noise.

**5. Which metric measures how many actual positive cases were identified?**

Accuracy

Precision

Recall

*

Recall measures the proportion of actual positives correctly identified
