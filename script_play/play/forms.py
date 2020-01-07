from django import forms

class InputText(forms.Form):
    text = forms.CharField(label='text', max_length=100)