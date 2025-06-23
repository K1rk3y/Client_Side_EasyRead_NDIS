from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import get_user_model
from .models import UploadedFile

User = get_user_model()


class PDFUploadForm(forms.ModelForm):
    class Meta:
        model = UploadedFile
        fields = ["file"]
        widgets = {
            "file": forms.FileInput(
                attrs={"accept": ".pdf,.docx", "class": "form-control"}
            )
        }

    def clean_file(self):
        file = self.cleaned_data.get("file")
        if file:
            if not file.name.lower().endswith((".pdf", ".docx")):
                raise forms.ValidationError("Only PDF or DOCX files are allowed.")
            if file.size > 100 * 1024 * 1024:  # 100MB
                raise forms.ValidationError("File size must be under 100MB.")
        return file


class CustomUserCreationForm(UserCreationForm):
    name = forms.CharField(max_length=100, required=True)
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ("username", "name", "email", "password1", "password2")

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data["email"]
        user.name = self.cleaned_data["name"]
        if commit:
            user.save()
        return user


class CustomAuthenticationForm(AuthenticationForm):
    username = forms.EmailField(
        widget=forms.EmailInput(attrs={"class": "form-control"})
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={"class": "form-control"})
    )
