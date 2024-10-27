# tense_classifier
This resporatory is an AI model working on transformer and tokenizer which can classsify the tenses in past, present and future tense. This is a pretrained model.
## TenseSense - Identify Sentence Tense

This program, TenseSense, is a graphical user interface (GUI) application that allows you to identify the tense of a sentence. It utilizes a pre-trained BERT model for text classification.

### Features:

* **Text Input:** Enter a sentence directly into the text entry field.
* **Voice Input:** Use your microphone to speak a sentence for analysis.
* **Tense Prediction:** The program predicts the tense (past, present, or future) of the entered sentence using a pre-trained model.
* **Feedback:** You can provide feedback on the prediction accuracy. 
* **(Optional):** Currently disabled, but the code includes a section for potentially incorporating user feedback to update the model.

### Requirements:

* Python 3.x
* `tkinter` library (included in most Python installations)
* `torch` library (for deep learning)
* `transformers` library (for BERT model)
* `speech_recognition` library (for voice input)
* `pyttsx3` library (for text-to-speech)

### Installation:

1. Install the required libraries using `pip`:

```bash
pip install tkinter torch transformers speech_recognition pyttsx3
```

2. Download a pre-trained BERT model (e.g., `bert-base-multilingual-cased`) and place it in the same directory as the script (or adjust the path in the code).


### Usage:

1. Run the script:

```bash
python tense_bot.py
```

2. The GUI window will appear.
3. Enter a sentence in the text entry field or click the "Speak" button to use voice input.
4. Click the "Submit" button to analyze the sentence.
5. The program will display the predicted tense and speak the result using text-to-speech.
6. You can provide feedback on the prediction by clicking "Yes" or "No" in the popup window.
 - If you select "No", you can optionally provide the correct tense in the following prompt. (Note: Model update functionality is currently disabled)
