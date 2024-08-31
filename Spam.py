#!/usr/bin/env python
# coding: utf-8

# In[3]:


from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from imblearn.over_sampling import SMOTE

def clean_text(msg):
    sp_words=stopwords.words('english')
    sp_words.remove('not')
    sp_words.remove("don't")
    sp_words.remove("didn't")
    sp_words.remove("hasn't")
    sp_words.remove("haven't")
    sp_words.remove("wasn't")
    sp_words.remove("weren't")

    def remove_punct(msg):
        return re.sub(f'[{string.punctuation}]','',msg)

    def remove_stopwds(msg):
        words=word_tokenize(msg)
        new_words=[]
        for w in words:
            if(w not in sp_words):
                new_words.append(w)
        return " ".join(new_words)

    def stemming(msg):
        ps=PorterStemmer()
        words=word_tokenize(msg)
        new_words=[]
        for w in words:
            new_words.append(ps.stem(w))
        return " ".join(new_words) 
    
    X1=remove_punct(msg)
    X2=X1.lower()
    X3=remove_stopwds(X2)
    X4=stemming(X3)
    return X4

df=pd.read_csv('C:/dataset/sentiment/spam_ham.txt',delimiter="\t",header=None,names=['msg_type','msg'])
df.msg=list(map(clean_text,df.msg))
cv=TfidfVectorizer(binary=False)
X=cv.fit_transform(df.msg).toarray()
y=df.msg_type
smote=SMOTE()
new_X,new_y=smote.fit_resample(X,y)
clf=MultinomialNB()
clf.fit(new_X,new_y)

win=Tk()
win.state("zoomed")
win.resizable(width=False,height=False)
win.configure(bg="orange")
win.title("My project")

lbl_title=Label(win,text="Spam Detection",font=('',50,'bold'),bg='orange')
lbl_title.place(relx=.3,rely=0)

def predict_single(entry_user,lbl_result):
    user_review=entry_user.get()
    ct=clean_text(user_review)
    X_test=cv.transform([ct]).toarray()
    pred=clf.predict(X_test)
    if(pred[0]=='ham'):
        lbl_result.configure(text="Ham",fg="white")
    else:
        lbl_result.configure(text="Spam",fg="red")

def predict_save(entry_src,entry_dest):        
    srcpath=entry_src.get()
    destpath=entry_dest.get()
    df=pd.read_csv(srcpath,header=None,names=['msg'])
    X=df.msg.map(clean_text)
    X_test=cv.transform(X).toarray()  
    pred=clf.predict(X_test)
    result_df=pd.DataFrame()
    result_df['msg']=df.msg
    result_df['msg_type']=pred
    result_df['msg_type']=result_df['msg_type']
    result_df.to_csv(destpath,index=False,sep="\t")
    messagebox.showinfo('Result',"Prediction Done...")
def logout():
    option=messagebox.askyesno('Confirmation','Do you want to logout?')
    if(option==True):
        home_screen()
    else:
        pass
def home_screen():
    frm=Frame(win,bg='green')
    frm.place(relx=0,rely=.15,relwidth=1,relheight=1)
    
    lbl_user=Label(frm,text="Username",font=('',20,'bold'),bg='green')
    lbl_user.place(relx=.3,rely=.3)

    entry_user=Entry(frm,font=('',20,'bold'),bd=10)
    entry_user.place(relx=.42,rely=.3)
    entry_user.focus()

    lbl_pass=Label(frm,text="Password",font=('',20,'bold'),bg='green')
    lbl_pass.place(relx=.3,rely=.4)

    entry_pass=Entry(frm,font=('',20,'bold'),bd=10,show="*")
    entry_pass.place(relx=.42,rely=.4)

    btn_login=Button(frm,command=lambda:welcome_screen(entry_user,entry_pass),text="login",font=('',20,'bold'),bd=10,width=10)
    btn_login.place(relx=.45,rely=.5)

def welcome_screen(entry_user=None,entry_pass=None):
    if(entry_user!=None and entry_pass!=None):
        user=entry_user.get()
        pwd=entry_pass.get()
    else:
        user="admin"
        pwd="admin"
    if(len(user)==0 or len(pwd)==0):
        messagebox.showwarning("validation","Please fill both fields")
        return
    else:
        if(user=="admin" or pwd=="admin"):
            frm=Frame(win,bg='green')
            frm.place(relx=0,rely=.15,relwidth=1,relheight=1)

            btn_single=Button(frm,command=lambda:single_feedback_screen(),text="Single Prediction",font=('',20,'bold'),bd=10,width=25)
            btn_single.place(relx=.35,rely=.2)

            btn_bulk=Button(frm,command=lambda:bulk_feedback_screen(),text="Bulk Prediction",font=('',20,'bold'),bd=10,width=25)
            btn_bulk.place(relx=.35,rely=.4)

            btn_logout=Button(frm,command=lambda:logout(),text="logout",font=('',20,'bold'),bd=10)
            btn_logout.place(relx=.9,rely=0)
        else:
            messagebox.showerror("Fail","Invalid Username/Password")
    
def single_feedback_screen():
    frm=Frame(win,bg='green')
    frm.place(relx=0,rely=.15,relwidth=1,relheight=1)
    
    lbl_user=Label(frm,text="Enter Msg:",font=('',20,'bold'),bg='green')
    lbl_user.place(relx=.28,rely=.3)

    entry_user=Entry(frm,font=('',20,'bold'),bd=10)
    entry_user.place(relx=.45,rely=.3)
    entry_user.focus()

    lbl_result=Label(frm,text="Prediction:",font=('',20,'bold'),bg='green')
    lbl_result.place(relx=.35,rely=.55)

    btn_login=Button(frm,command=lambda:predict_single(entry_user,lbl_result),text="predict",font=('',20,'bold'),bd=10,width=8)
    btn_login.place(relx=.47,rely=.42)

    btn_back=Button(frm,command=lambda:welcome_screen(),text="back",font=('',20,'bold'),bd=10)
    btn_back.place(relx=.9,rely=0)

def bulk_feedback_screen():
    frm=Frame(win,bg='green')
    frm.place(relx=0,rely=.15,relwidth=1,relheight=1)
    
    lbl_src=Label(frm,text="Select Source file:",font=('',20,'bold'),bg='green')
    lbl_src.place(relx=.26,rely=.2)
    
    lbl_dest=Label(frm,text="Select Destination Directory:",font=('',20,'bold'),bg='green')
    lbl_dest.place(relx=.26,rely=.32)
    

    entry_src=Entry(frm,font=('',20,'bold'),bd=10)
    entry_src.place(relx=.45,rely=.2)
    entry_src.focus()

    entry_dest=Entry(frm,font=('',20,'bold'),bd=10)
    entry_dest.place(relx=.45,rely=.32)
    
    btn_browse=Button(frm,command=lambda:browse(entry_src),text="browse",font=('',20,'bold'),bd=10,width=8)
    btn_browse.place(relx=.7,rely=.2)

    btn_browse2=Button(frm,command=lambda:browse2(entry_dest),text="browse",font=('',20,'bold'),bd=10,width=8)
    btn_browse2.place(relx=.7,rely=.32)
    
    btn_login=Button(frm,command=lambda:predict_save(entry_src,entry_dest),text="predict and save",font=('',20,'bold'),bd=10,width=15)
    btn_login.place(relx=.47,rely=.6)

    btn_back=Button(frm,command=lambda:welcome_screen(),text="back",font=('',20,'bold'),bd=10)
    btn_back.place(relx=.9,rely=0)
    
def browse(entry_path):
    file_path=filedialog.askopenfilename()
    entry_path.delete(0,END)
    entry_path.insert(0,file_path)

def browse2(entry_path):
    file_path=filedialog.askdirectory()+"/result.txt"
    entry_path.delete(0,END)
    entry_path.insert(0,file_path)    
home_screen()    
win.mainloop() #to make window visible


# In[ ]:




