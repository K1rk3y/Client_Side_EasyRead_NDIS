from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, StringField, PasswordField
from wtforms.validators import DataRequired, Regexp, Email, EqualTo, Length
from flask_wtf.file import FileAllowed, FileRequired


class PDFUploadForm(FlaskForm):
    pdf_file = FileField(
        "Upload File",
        validators=[
            FileRequired(message="Please select a file."),
            FileAllowed(["pdf", "docx"], "Only PDF or DOCX files are allowed."),
        ],
    )
    submit = SubmitField("Submit")


class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Login")


class SignupForm(FlaskForm):
    name = StringField("Name", validators=[DataRequired()])
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField(
        "Confirm Password", validators=[DataRequired(), EqualTo("password")]
    )
    submit = SubmitField("Sign Up")
