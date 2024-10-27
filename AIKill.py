import tkinter as tk
from tkinter import simpledialog, messagebox
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import speech_recognition as sr
import pyttsx3

MAX_LEN = 32
MODEL_SAVE_PATH = 'tense_classifier.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3).to(DEVICE)

def load_model(filepath=MODEL_SAVE_PATH):
    try:
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {filepath}.")
    except FileNotFoundError:
        print(f"No checkpoint found at {filepath}. Starting from scratch.")

load_model()

label_map_reverse = {0: "Past", 1: "Present", 2: "Future"}

engine = pyttsx3.init()
class TenseBotGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TenseSense")
        self.geometry("600x300")
        
        self.label = tk.Label(self, text="Enter a sentence:")
        self.label.pack(pady=10)
        
        self.text_entry = tk.Entry(self, width=60)
        self.text_entry.pack(pady=5)
        
        self.submit_button = tk.Button(self, text="Submit", command=self.predict_tense)
        self.submit_button.pack(pady=10)
        
        self.voice_button = tk.Button(self, text="Speak", command=self.voice_typing)
        self.voice_button.pack(pady=5)
        
        self.quit_button = tk.Button(self, text="Exit", command=self.quit_program)
        self.quit_button.pack(pady=10)
        
        self.result_label = tk.Label(self, text="")
        self.result_label.pack(pady=10)
        
    def predict_tense(self):
        sentence = self.text_entry.get()
        
        encoding = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=MAX_LEN,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)

        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, prediction = torch.max(outputs.logits, dim=1)
        
        predicted_tense = label_map_reverse[prediction.item()]
        result_text = f"The sentence is in the {predicted_tense} tense."
        self.result_label.config(text=result_text)
        engine.say(result_text)
        engine.runAndWait()
        
        feedback = messagebox.askquestion("Feedback", "Is this correct?")
        if feedback == 'no':
            correct_label = simpledialog.askstring("Correction", "Please type the correct tense (Past/Present/Future):")
            if correct_label in label_map_reverse.values():
                correct_label_idx = list(label_map_reverse.values()).index(correct_label)
                correct_label_tensor = torch.tensor([correct_label_idx]).to(DEVICE)
                # Here you can update the model if needed
                # For now, we'll just print a message
                print(f"Correct label received: {correct_label}, but training is disabled.")
            else:
                messagebox.showerror("Error", "Invalid tense provided. Please enter Past, Present, or Future.")
        print()

    def voice_typing(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            self.result_label.config(text="Listening...")
            engine.say("Listening...")
            engine.runAndWait()
            audio = recognizer.listen(source)
        
        try:
            sentence = recognizer.recognize_google(audio)
            self.text_entry.delete(0, tk.END)
            self.text_entry.insert(0, sentence)
            self.predict_tense()
        except sr.UnknownValueError:
            messagebox.showerror("Error", "Sorry, I did not understand the audio.")
        except sr.RequestError:
            messagebox.showerror("Error", "Sorry, there was an issue with the speech recognition service.")

    def quit_program(self):
        messagebox.showinfo("Thank You", "Made by Suhani and Tanish")
        self.destroy()

if __name__ == "__main__":
    app = TenseBotGUI()
    app.mainloop()
