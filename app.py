import customtkinter as ctk
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
# Creating main customtkinter window
ctk.set_default_color_theme("green")
root = ctk.CTk()
root.title("Toxic Comment Analysis")
root.geometry("1000x600")
from textblob import TextBlob

root.resizable(False, False)
# Create and configure widgets and packs
frame = ctk.CTkFrame(master=root)
frame.pack(pady=20, padx=20, fill="both", expand=True)

# Create Textbox
textbox = ctk.CTkTextbox(master=frame, height=415, width=850, activate_scrollbars=True)

textbox.insert("0.0", "text for analysis")  # insert at line 0 character 0
textbox.delete("0.0", "end")  # delete all text
textbox.configure(state="normal")  # configure textbox to be read-only
textbox.pack(pady=20, padx=10)


# Text analysis function
def click_result():
    analysis = textbox.get("1.0", "end")
    blob = TextBlob(analysis)
    sentiment = blob.sentiment.polarity
    if sentiment > 0.0:
        result_label.configure(text=f"The text is generally non-toxic.")
    elif sentiment < 0.0:
        result_label.configure(text=f"The text is generally toxic.")
    else:
        result_label.configure(text=f"The text is neutral.")


# Create analysis button
button = ctk.CTkButton(master=frame, height=30, width=80, text="Analyse", font=("Arial", 16), command=click_result)
button.pack(pady=5, padx=0)

result_label = ctk.CTkLabel(master=frame, text="", font=("Arial", 18))
result_label.pack(pady=10, padx=10)

root.mainloop()