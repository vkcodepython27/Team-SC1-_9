from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

class ChronicCare(App):
    def build(self):
        #Input Fields
        self.middle_layout=GridLayout(cols=3)
        self.layout=BoxLayout(orientation='vertical')
        self.age_input=TextInput(hint_text='Enter Your Age')
        #Test Input Buttons
        self.test_result=0
        self.test_normal_btn=Button(text='Normal')
        self.test_abnormal_btn=Button(text='Abnormal')
        self.test_inconclusive_btn=Button(text='Inconclusive')
        self.test_normal_btn.bind(on_press=lambda x:self.set_test_result(0))
        self.test_abnormal_btn.bind(on_press=lambda x:self.set_test_result(1))
        self.test_inconclusive_btn.bind(on_press=lambda x:self.set_test_result(2))
        self.middle_layout.add_widget(self.test_normal_btn)
        self.middle_layout.add_widget(self.test_abnormal_btn)
        self.middle_layout.add_widget(self.test_inconclusive_btn)
        #Number of Days Stayed
        self.length_of_stay=TextInput(hint_text='Enter the number of days of Stay:')
        #Number of Times Admitted Input
        self.admitted_input=TextInput(hint_text='Enter the number of time Admitted:')
        #Submit Button Widget
        self.submit_btn=Button(text='Predict:')
        self.submit_btn.bind(on_press=self.predict_result)
        #Add Everything To the Layout
        self.layout.add_widget(self.age_input)
        self.layout.add_widget(self.middle_layout)
        self.layout.add_widget(self.length_of_stay)
        self.layout.add_widget(self.admitted_input)
        self.layout.add_widget(self.submit_btn)
        self.refresh_btn=Button(text='Refresh Menu!')
        self.refresh_btn.bind(on_press=self.refresh_app)
        self.layout.add_widget(self.refresh_btn)
        return self.layout
    def refresh_app(self,instance):
        self.stop()
        ChronicCare().run()
    def set_test_result(self,value):
        self.test_result=value
    def predict_result(self,instance):
        #Load and preprocess dataset
        df=pd.read_csv('healthcare_dataset.csv')
        test_results_mapping={'Normal':0,'Inconclusive':2,'Abnormal':1}
        df['Test_Results_Encoded']=df['Test Results'].map(test_results_mapping)
        df['Date of Admission']=pd.to_datetime(df['Date of Admission'])
        df['Discharge Date']=pd.to_datetime(df['Discharge Date'])
        df['Length of Stay']=(df['Discharge Date']-df['Date of Admission']).dt.days
        df['Name']=df['Name'].str.lower()
        name_count=df['Name'].value_counts()
        df['frequency_of_times_admitted']=df['Name'].map(name_count)
        df['Readmission']=df['frequency_of_times_admitted'].apply(lambda x:1 if x>1 else 0)
        #Prepare features and labels
        x=df.drop(['Name','Gender','Blood Type','Medical Condition','Date of Admission','Doctor','Hospital','Insurance Provider','Billing Amount','Room Number','Admission Type','Discharge Date','Medication','Test Results','Readmission'],axis=1)
        y=df['Readmission']
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
        model=KNeighborsClassifier()
        model.fit(x_train,y_train)
        #Take user input from app
        try:
            age=int(self.age_input.text)
            stay=int(self.length_of_stay.text)
            freq=int(self.admitted_input.text)
            result=model.predict([[age,self.test_result,stay,freq]])
            status='Readmission Likely'if result[0]==1 else'No Readmission Risk'
        except:
            status='Invalid input! Please enter valid numbers.'
        #Display result
        self.layout.add_widget(Label(text=f'Prediction:{status}'))

ChronicCare().run()
