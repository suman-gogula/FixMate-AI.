# FixMate AI - Assignment and Examination Notes

## 1. Project Title
**FixMate AI - Smart Maintenance Ticket Classifier Using Natural Language Processing**

## 2. Abstract
Maintenance complaints in hostels, colleges and offices are written in free English text and are usually sorted by hand, which is slow and error-prone. FixMate AI reads a complaint, cleans it with an NLP pipeline (lowercasing, punctuation removal, tokenization, stop-word removal, stemming), converts it to numbers with TF-IDF and predicts its category using Logistic Regression. A separate rule-based module decides urgency (High, Medium, Low), then the complaint is assigned to a department with a recommended action. The system is built with Python, NLTK, Scikit-learn and Streamlit and runs fully offline.

## 3. Introduction
Natural Language Processing lets computers work with human language. Organizations receive hundreds of complaints written in different words. FixMate AI shows how NLP and Machine Learning can automatically understand such text and route it to the right team.

## 4. Problem Statement
Manual sorting of maintenance complaints causes delays, wrong department assignment and missed emergencies (for example gas leakage). A system is needed that understands normal English complaints and automatically predicts category, urgency, department and action.

## 5. Objectives
1. Accept complaints in normal English.
2. Clean text using an NLP pipeline.
3. Predict the complaint category using ML.
4. Detect urgency using transparent rules.
5. Assign the department and recommend an action.
6. Show history, analytics, accuracy and confusion matrix.

## 6. Existing System
Complaints are written on paper, registers, phone calls or emails; a supervisor reads each one and decides the team and priority. This is slow, inconsistent, depends on one person, and urgent problems can be missed.

## 7. Proposed System
An automatic classifier: the user types a complaint; the system preprocesses it, predicts category (ML), urgency (rules), department (lookup) and action, then stores it in history with analytics. It is fast, consistent and works 24x7.

## 8. Technologies Used
| Tool | Use |
|---|---|
| Python 3 | Programming language |
| Streamlit | Web user interface |
| NLTK | Stemmer, tokenizer, stop-word list |
| Scikit-learn | TF-IDF, Logistic Regression, accuracy, confusion matrix |
| Pandas, NumPy | Data handling and numbers |
| Matplotlib | Charts |

## 9. NLP Concepts (simple English)
- **What is NLP?** Natural Language Processing is the branch of AI that helps computers read, understand and process human language such as English.
- **Tokenization:** Breaking a sentence into small pieces (tokens), usually words. "Water is leaking" -> [water, is, leaking].
- **Stop words:** Very common words (is, the, in, my) that add little meaning. We remove them so the model focuses on important words. We keep "not" because "not working" changes the meaning.
- **Stemming:** Cutting a word to its root form: leaking -> leak, working -> work. Different forms of a word are then treated as one word. (Lemmatization does the same using a dictionary; stemming is simpler and needs no downloads.)
- **TF-IDF:** Converts text into numbers. *TF (Term Frequency)* = how often a word appears in a complaint. *IDF (Inverse Document Frequency)* = how rare the word is across all complaints. Common words get low weight; distinctive words like "leak" or "bulb" get high weight. Score = TF x IDF.
- **Text classification:** Putting a piece of text into one of several predefined classes (here, 6 categories).
- **Logistic Regression:** A classification algorithm. It learns a weight for every word from training data, combines them, and uses the softmax function to give a probability for each category. The category with the highest probability is the prediction. Despite "regression" in its name, it is used for classification.
- **Why machine learning here?** People describe the same problem in many ways ("tap dripping", "water falling from faucet"). Writing rules for every sentence is impossible; ML learns patterns from examples and handles new wording.
- **Why rules for urgency?** Urgency depends on a few critical keywords (fire, gas, flooding). Rules are simple, safe, transparent and easy to explain. **Urgency detection is a separate component from the ML category classifier.**

## 10. System Architecture
```
User complaint
     |
     v
[ NLP Preprocessing ] lowercase -> remove punctuation -> tokenize -> remove stop words -> stem
     |                                             |
     v                                             v
[ TF-IDF Vectorizer ]                     [ Rule-based Urgency ]  (uses the original text)
     |                                             |
     v                                             |
[ Logistic Regression ] -> Category                |
     |                                             |
     +--> Department (lookup table)                |
     +--> Recommended action  <--------------------+
     |
     v
Result display + History (CSV) + Analytics charts
```

## 11. Working Methodology
1. Create `dataset.csv` (211 labelled complaints).
2. Preprocess every complaint.
3. Split data: 80% training, 20% testing (stratified, fixed random seed).
4. Fit TF-IDF and Logistic Regression on the training set only.
5. Calculate accuracy and confusion matrix on the test set.
6. For a new complaint: preprocess -> predict category with confidence -> apply urgency rules -> assign department -> choose action.
7. Save to history and update dashboard.

## 12. Algorithm
```
INPUT: complaint text
1. text  = lowercase(text)
2. text  = remove punctuation and digits
3. tokens = split text into words
4. tokens = remove stop words
5. tokens = stem each token
6. vector = TF-IDF(tokens)
7. category, confidence = LogisticRegression.predict(vector)
8. urgency = first matching rule: High rules, then Medium rules, then Low rules, else category default
9. department = lookup(category)
10. action = rule(category, urgency)
OUTPUT: category, confidence, urgency, department, action
```

## 13. Two Practical Examples
**Example 1**
- Input: *Water is leaking continuously from my bathroom ceiling.*
- Lowercase/punctuation removed: `water is leaking continuously from my bathroom ceiling`
- Tokens after stop-word removal: `water, leaking, continuously, bathroom, ceiling`
- After stemming: `water leak continu bathroom ceil`
- Category: **Plumbing** (words like leak/water/bathroom are strongly linked to plumbing in the training data)
- Urgency: **High** (rule: "leak" + "continuously/ceiling" = major water leakage)
- Department: **Plumbing Department**
- Action: **Immediate inspection and repair.**

**Example 2**
- Input: *The light bulb in my bedroom is not working.*
- Tokens after stop-word removal: `light, bulb, bedroom, not, working`
- After stemming: `light bulb bedroom not work`
- Category: **Electrical**
- Urgency: **Low** (rule: light bulb replacement)
- Department: **Electrical Department**
- Action: **Schedule a light bulb replacement.**

(The exact confidence percentage is shown by the app when you run it - quote that number, not one from these notes.)

## 14. Advantages
Fast and consistent; works offline and free; easy to understand; urgency decisions are transparent; shows accuracy honestly; easy to extend with more data.

## 15. Limitations
- Small dataset (211 complaints), so accuracy is limited and can vary.
- Only English; spelling mistakes are not corrected.
- Urgency rules use keywords, so unusual wording can be missed.
- One category per complaint (a complaint mentioning two problems gets one label).
- Bag-of-words/TF-IDF does not fully understand context or meaning.

## 16. Real-World Applications
Hostel and college maintenance desks, apartment and facility management, hospital and hotel maintenance, smart-city complaint portals, customer-support ticket routing, IT helpdesks.

## 17. Future Enhancements
Larger real dataset; spelling correction; Hindi and other languages; ML-based urgency prediction; deep learning models such as BERT; email/WhatsApp integration; technician assignment and SLA tracking; database and login system.

## 18. Conclusion
FixMate AI demonstrates a complete NLP + ML pipeline - preprocessing, TF-IDF, Logistic Regression - combined with transparent rules to automate maintenance ticket handling. It is practical, easy to explain and can be improved with more data.

---

# Viva Preparation - 15 Questions

1. **What is NLP?** The field of AI that lets computers understand and process human language.
2. **Which libraries did you use and why?** Streamlit (UI), NLTK (text preprocessing), Scikit-learn (TF-IDF and model), Pandas/NumPy (data), Matplotlib (charts).
3. **What is tokenization?** Splitting text into words (tokens).
4. **What are stop words? Why remove them?** Common words like "is" and "the". Removing them reduces noise and keeps important words.
5. **Why did you keep "not"?** Because "not working" has the opposite meaning of "working".
6. **Stemming vs lemmatization?** Stemming chops word endings (fast, rough); lemmatization uses a dictionary to give a real word (slower, more accurate). I used the Porter stemmer.
7. **What is TF-IDF?** A number for each word = how frequent it is in the complaint x how rare it is across all complaints.
8. **Why not just count words?** Counting gives common words too much importance; TF-IDF reduces their weight.
9. **Which algorithm did you use for classification?** Logistic Regression - it gives a probability for each category and picks the highest.
10. **Why machine learning?** Users phrase the same problem differently; ML learns patterns from examples instead of hand-writing endless rules.
11. **How did you train the model?** 80% of the dataset is used for training; the model learns word weights for each category.
12. **How is accuracy calculated? Is it fake?** No. The model predicts the 20% test data it never saw; accuracy = correct predictions / total test predictions. The app calculates it live.
13. **What is a confusion matrix?** A table of actual vs predicted categories; the diagonal shows correct predictions, other cells show mistakes.
14. **Is urgency detected by ML?** No. It is a separate rule-based component using keywords (fire, gas leakage, flooding...), so it is transparent and safe.
15. **Real-world applications and limits?** Hostel/facility helpdesks, ticket routing in support systems. Limits: small dataset, English only, keyword-based urgency.

**Bonus answers:** *Why a train-test split?* To check the model on data it has not seen (avoid fooling ourselves). *What if the complaint is unknown?* The confidence will be low and the app shows a warning. *How to improve?* Add more labelled complaints.
