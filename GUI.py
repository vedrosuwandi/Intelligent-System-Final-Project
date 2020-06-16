import Chatbot

import tkinter as tk
from tkinter import *


def sent():
    messages = message.get()
    reply = Chatbot.chat(messages)

    if messages != "":
        chatpanel.config(state=NORMAL)
        chatpanel.insert(END, "You: " + messages + '\n')
        chatpanel.insert(END, "RelaxnChatBot: " + reply + '\n')
        enterchat.delete(0, 'end')
        chatpanel.config(state=DISABLED)
        chatpanel.yview(END)
    else:
        return

window = tk.Tk()
window.title("ChatBot")
window.geometry("400x440")
window.resizable(width='false', height='false')

message = StringVar()
chatpanel = tk.Text(window, bg="white", height="8", width="50", font="Arial")
chatpanel.config(state="disable")
enterchat = tk.Entry(window, bd=0, bg="white", font="Arial", textvariable=message)
Send = tk.Button(window, text="Send", bd=0, bg="lightgreen", command=sent)

scrollbar = Scrollbar(window, command=chatpanel.yview, cursor="heart")
chatpanel['yscrollcommand'] = scrollbar.set

scrollbar.place(x=378,y=6, height=386)
chatpanel.place(x=6, y=6, height=386, width=390)
enterchat.place(x=6, y=401, height=30, width=280)
Send.place(x=295, y=401, height=30 , width=100)


window.mainloop()


