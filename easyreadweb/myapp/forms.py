from django import forms

class LoginForm(forms.Form):
    email = forms.EmailField(
        label='Email',
        max_length=254,
        widget=forms.EmailInput(attrs={'class': 'form-input', 'size': 32})
    )
    password = forms.CharField(
        label='Password',
        widget=forms.PasswordInput(attrs={'class': 'form-input', 'size': 32})
    )

class RegisterForm(forms.Form):
    name = forms.CharField(
        label='Name',
        max_length=100,
        widget=forms.TextInput(attrs={'class': 'form-input', 'size': 32})
    )
    email = forms.EmailField(
        label='Email',
        max_length=254,
        widget=forms.EmailInput(attrs={'class': 'form-input', 'size': 32})
    )
    password = forms.CharField(
        label='Password',
        widget=forms.PasswordInput(attrs={'class': 'form-input', 'size': 32})
    )
    confirm_password = forms.CharField(
        label='Confirm Password',
        widget=forms.PasswordInput(attrs={'class': 'form-input', 'size': 32})
    )

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get('password')
        confirm_password = cleaned_data.get('confirm_password')
        if password and confirm_password and password != confirm_password:
            self.add_error('confirm_password', 'Passwords do not match.')
        return cleaned_data

class PDFUploadForm(forms.Form):
    pdf_file = forms.FileField(
        label='Upload File',
        widget=forms.ClearableFileInput(attrs={
            'class': 'form-input',
            'accept': '.pdf,.docx',
            'id': 'pdfInput',
        })
    )
    submit = forms.CharField(widget=forms.HiddenInput(), required=False)
