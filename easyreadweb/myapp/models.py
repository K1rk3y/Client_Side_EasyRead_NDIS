from django.contrib.auth.models import AbstractUser
from django.db import models

class CustomUser(AbstractUser):
    # Add extra fields here if needed, e.g. phone = models.CharField(max_length=20, blank=True)
    role = models.CharField(max_length=50, blank=True, default="user")
    profilepic = models.TextField(blank=True, null=True, help_text="Base64-encoded profile picture")
    AItoken = models.TextField(blank=True, null=True, help_text="OpenAI token for this user")

class ContactMessage(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    message = models.TextField()
    submitted_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} ({self.email}) at {self.submitted_at}"
